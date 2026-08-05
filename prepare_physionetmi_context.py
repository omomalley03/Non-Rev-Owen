"""Build PhysioNetMI windows with real EDF context around the old LMDB windows.

The existing PhysioNetMI cache is the behavioral source of truth for the core
``(64, 800)`` windows.  This script reads the same LMDB keys, places each old
window exactly in the center, and fills pre/post context from the raw EDF run.

The old cache appears to use 5 s epochs starting 0.5 s before the task cue, so
the default raw context alignment is ``event_onset - 80`` samples at 160 Hz.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import lmdb
import numpy as np
import numpy.core as _np_core
import numpy.core.multiarray as _np_multiarray
import numpy.core.numeric as _np_numeric
from scipy import signal


sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
sys.modules.setdefault("numpy._core.numeric", _np_numeric)

KEY_RE = re.compile(r"^S(?P<subject>\d{3})R(?P<run>\d{2})-(?P<annotation>\d+)$")
MOTOR_RUNS = {4, 6, 8, 10, 12, 14}
RUN_LABELS = {
    4: {"T1": 0, "T2": 1},
    8: {"T1": 0, "T2": 1},
    12: {"T1": 0, "T2": 1},
    6: {"T1": 2, "T2": 3},
    10: {"T1": 2, "T2": 3},
    14: {"T1": 2, "T2": 3},
}


def _to_key_bytes(key: str | bytes) -> bytes:
    return key if isinstance(key, bytes) else str(key).encode()


def _parse_splits(value: str) -> list[str]:
    splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("at least one split is required")
    return splits


def _parse_key(key: str) -> tuple[int, int, int]:
    match = KEY_RE.match(key)
    if match is None:
        raise ValueError(f"unexpected LMDB key format: {key!r}")
    return (
        int(match.group("subject")),
        int(match.group("run")),
        int(match.group("annotation")),
    )


def _read_edf(path: Path) -> tuple[np.ndarray, list[tuple[float, float, str]], float]:
    data = path.read_bytes()
    header_bytes = int(data[184:192])
    n_records = int(data[236:244])
    record_duration_s = float(data[244:252])
    n_signals = int(data[252:256])

    offset = 256
    fields: list[list[str]] = []
    for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
        fields.append(
            [
                data[offset + idx * width : offset + (idx + 1) * width]
                .decode("latin1")
                .strip()
                for idx in range(n_signals)
            ]
        )
        offset += n_signals * width

    labels = fields[0]
    samples_per_record = np.array([int(value) for value in fields[8]], dtype=np.int64)
    if labels[-1] != "EDF Annotations":
        raise ValueError(f"{path}: expected final EDF annotation signal, got {labels[-1]!r}")
    if len(set(samples_per_record[:-1].tolist())) != 1:
        raise ValueError(f"{path}: EEG channels have unequal samples per record")

    raw = np.frombuffer(data, dtype="<i2", offset=header_bytes)
    expected = int(n_records * samples_per_record.sum())
    if raw.size != expected:
        raise ValueError(f"{path}: expected {expected} data samples, got {raw.size}")

    by_record = raw.reshape(n_records, int(samples_per_record.sum()))
    starts = np.concatenate(([0], np.cumsum(samples_per_record)))
    eeg = np.stack(
        [
            by_record[:, starts[idx] : starts[idx + 1]].reshape(-1)
            for idx in range(n_signals - 1)
        ]
    ).astype(np.float32, copy=False)
    eeg = eeg - eeg.mean(axis=0, keepdims=True)

    ann_bytes = by_record[:, starts[-2] : starts[-1]].astype("<i2").tobytes()
    ann_text = ann_bytes.decode("latin1", errors="ignore")
    annotations = _parse_edf_annotations(ann_text)
    sample_rate_hz = float(samples_per_record[0]) / record_duration_s
    return np.ascontiguousarray(eeg, dtype=np.float32), annotations, sample_rate_hz


def _parse_edf_annotations(text: str) -> list[tuple[float, float, str]]:
    annotations: list[tuple[float, float, str]] = []
    for tal in text.split("\x00"):
        if not tal:
            continue
        parts = tal.split("\x14")
        if not parts or not parts[0]:
            continue
        match = re.match(
            r"([+-]?\d+(?:\.\d+)?)(?:\x15(\d+(?:\.\d+)?))?",
            parts[0],
        )
        if match is None:
            continue
        labels = [part for part in parts[1:] if part]
        for label in labels:
            annotations.append(
                (
                    float(match.group(1)),
                    float(match.group(2) or 0.0),
                    label,
                )
            )
    return annotations


class EdfRunCache:
    def __init__(self, raw_root: Path, target_fs: float, max_runs: int = 8) -> None:
        self.raw_root = raw_root
        self.target_fs = float(target_fs)
        self.max_runs = max_runs
        self.cache: OrderedDict[tuple[int, int], tuple[np.ndarray, list[tuple[float, float, str]], float]] = (
            OrderedDict()
        )

    def get(self, subject: int, run: int) -> tuple[np.ndarray, list[tuple[float, float, str]], float]:
        cache_key = (subject, run)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        path = self.raw_root / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"
        if not path.exists():
            raise FileNotFoundError(path)
        eeg, annotations, sample_rate_hz = _read_edf(path)
        if sample_rate_hz != self.target_fs:
            ratio = Fraction(self.target_fs / sample_rate_hz).limit_denominator(1000)
            eeg = signal.resample_poly(eeg, ratio.numerator, ratio.denominator, axis=-1)
            eeg = np.ascontiguousarray(eeg, dtype=np.float32)
            sample_rate_hz = self.target_fs
        value = (eeg, annotations, sample_rate_hz)
        self.cache[cache_key] = value
        self.cache.move_to_end(cache_key)
        while len(self.cache) > self.max_runs:
            self.cache.popitem(last=False)
        return value


def _extract_with_edge_pad(
    eeg: np.ndarray,
    start: int,
    length: int,
) -> tuple[np.ndarray, int, int]:
    left_pad = max(0, -start)
    right_pad = max(0, start + length - eeg.shape[1])
    data_start = max(0, start)
    data_stop = min(eeg.shape[1], start + length)

    segment = eeg[:, data_start:data_stop]
    if left_pad or right_pad:
        segment = np.pad(segment, ((0, 0), (left_pad, right_pad)), mode="edge")
    if segment.shape != (eeg.shape[0], length):
        raise RuntimeError(f"unexpected padded segment shape {segment.shape}")
    return np.ascontiguousarray(segment, dtype=np.float32), left_pad, right_pad


def _annotation_label(run: int, annotation: tuple[float, float, str]) -> int | None:
    _, _, label = annotation
    return RUN_LABELS.get(run, {}).get(label)


def build_context_cache(
    raw_root: Path,
    processed_dir: Path,
    out_path: Path,
    splits: list[str],
    context_bins: int,
    core_len: int,
    event_start_offset: int,
    target_fs: float,
    max_records: int,
    labels_out: Path | None,
    subjects_out: Path | None,
    keys_out: Path | None,
    drop_edge_padded: bool,
) -> dict:
    env = lmdb.open(
        str(processed_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    run_cache = EdfRunCache(raw_root, target_fs=target_fs)

    try:
        with env.begin(write=False) as txn:
            split_keys = pickle.loads(txn.get(b"__keys__"))
            split_to_keys: dict[str, list[str]] = {}
            original_split_counts: dict[str, int] = {}
            for split in splits:
                if split not in split_keys:
                    available = ", ".join(sorted(split_keys))
                    raise KeyError(f"unknown split {split!r}; available: {available}")
                selected = [str(key) for key in split_keys[split]]
                if max_records > 0:
                    remaining = max_records - sum(len(v) for v in split_to_keys.values())
                    selected = selected[: max(0, remaining)]
                split_to_keys[split] = selected
                original_split_counts[split] = len(selected)

            dropped_edge_keys: list[str] = []
            split_counts: dict[str, int] = {}
            kept_by_split: dict[str, list[str]] = {}
            for split, selected in split_to_keys.items():
                kept: list[str] = []
                for key in selected:
                    subject, run, annotation_idx = _parse_key(key)
                    eeg, annotations, sample_rate_hz = run_cache.get(subject, run)
                    if annotation_idx >= len(annotations):
                        raise ValueError(
                            f"{key}: annotation index {annotation_idx} out of range "
                            f"for {len(annotations)} annotations"
                        )
                    annotation = annotations[annotation_idx]
                    cue_sample = int(round(annotation[0] * sample_rate_hz))
                    raw_start = cue_sample + event_start_offset - context_bins
                    left_pad = max(0, -raw_start)
                    right_pad = max(0, raw_start + core_len + 2 * context_bins - eeg.shape[1])
                    if drop_edge_padded and (left_pad or right_pad):
                        dropped_edge_keys.append(key)
                    else:
                        kept.append(key)
                kept_by_split[split] = kept
                split_counts[split] = len(kept)

            keys = [key for split in splits for key in kept_by_split[split]]

            if not keys:
                raise ValueError("no records selected")

            first = pickle.loads(txn.get(_to_key_bytes(keys[0])))
            first_sample = np.asarray(first["sample"], dtype=np.float32)
            if first_sample.shape != (64, core_len):
                raise ValueError(
                    f"expected old LMDB samples to have shape (64, {core_len}), "
                    f"got {first_sample.shape}"
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out = np.lib.format.open_memmap(
                out_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(keys), 64, core_len + 2 * context_bins),
            )
            labels = np.empty(len(keys), dtype=np.int64)
            subjects = np.empty(len(keys), dtype=np.int64)

            label_mismatches: list[dict] = []
            edge_padded_records = 0
            total_left_edge_pad = 0
            total_right_edge_pad = 0
            fs_values: set[float] = set()

            for idx, key in enumerate(keys):
                subject, run, annotation_idx = _parse_key(key)
                if run not in MOTOR_RUNS:
                    raise ValueError(f"{key}: expected motor imagery run, got R{run:02d}")

                record = pickle.loads(txn.get(_to_key_bytes(key)))
                center = np.asarray(record["sample"], dtype=np.float32)
                if center.shape != first_sample.shape:
                    raise ValueError(f"{key}: expected {first_sample.shape}, got {center.shape}")

                eeg, annotations, sample_rate_hz = run_cache.get(subject, run)
                fs_values.add(sample_rate_hz)
                if annotation_idx >= len(annotations):
                    raise ValueError(
                        f"{key}: annotation index {annotation_idx} out of range "
                        f"for {len(annotations)} annotations"
                    )
                annotation = annotations[annotation_idx]
                inferred_label = _annotation_label(run, annotation)
                old_label = int(record["label"])
                if inferred_label is not None and inferred_label != old_label:
                    label_mismatches.append(
                        {
                            "key": key,
                            "annotation": annotation,
                            "old_label": old_label,
                            "inferred_label": inferred_label,
                        }
                    )

                cue_sample = int(round(annotation[0] * sample_rate_hz))
                raw_start = cue_sample + event_start_offset - context_bins
                segment, left_pad, right_pad = _extract_with_edge_pad(
                    eeg,
                    raw_start,
                    core_len + 2 * context_bins,
                )
                if left_pad or right_pad:
                    edge_padded_records += 1
                    total_left_edge_pad += left_pad
                    total_right_edge_pad += right_pad

                segment[:, context_bins : context_bins + core_len] = center
                out[idx] = segment
                labels[idx] = old_label
                subjects[idx] = int(record["subject_id"])

                if (idx + 1) % 500 == 0 or idx + 1 == len(keys):
                    print(f"wrote {idx + 1}/{len(keys)} windows")

            out.flush()
    finally:
        env.close()

    if label_mismatches:
        examples = label_mismatches[:5]
        raise ValueError(f"label mismatches against EDF annotations: {examples}")

    if labels_out is not None:
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_out, labels)
    if subjects_out is not None:
        subjects_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(subjects_out, subjects)
    if keys_out is not None:
        keys_out.parent.mkdir(parents=True, exist_ok=True)
        keys_out.write_text("\n".join(keys) + "\n")

    sidecar = {
        "raw_root": str(raw_root),
        "processed_dir": str(processed_dir),
        "out": str(out_path),
        "splits": splits,
        "original_split_counts": original_split_counts,
        "split_counts": split_counts,
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "context_bins": context_bins,
        "core_len": core_len,
        "event_start_offset_samples": event_start_offset,
        "event_start_offset_seconds": event_start_offset / target_fs,
        "target_sample_rate_hz": target_fs,
        "sample_rate_hz_after_resample": sorted(fs_values),
        "center_source": "old LMDB sample copied exactly",
        "raw_context_preprocess": "common average reference across 64 EDF EEG channels",
        "labels_out": str(labels_out) if labels_out is not None else None,
        "subjects_out": str(subjects_out) if subjects_out is not None else None,
        "keys_out": str(keys_out) if keys_out is not None else None,
        "edge_padded_records": edge_padded_records,
        "drop_edge_padded": drop_edge_padded,
        "dropped_edge_padded_records": len(dropped_edge_keys),
        "dropped_edge_padded_keys": dropped_edge_keys,
        "total_left_edge_pad_samples": total_left_edge_pad,
        "total_right_edge_pad_samples": total_right_edge_pad,
    }
    sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps(sidecar, indent=2))
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("physionetmi_raw/files"),
        help="Unpacked PhysioNetMI raw root containing S001/S001R*.edf files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/home/omo26/data_owen/PhysioNetMI/processed"),
        help="Existing processed LMDB directory whose center windows are authoritative.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/physionetmi_train_val_context30.npy"),
        help="Output .npy path for padded windows.",
    )
    parser.add_argument(
        "--splits",
        type=_parse_splits,
        default=["train", "val"],
        help="Comma-separated LMDB splits to export, e.g. train,val,test.",
    )
    parser.add_argument(
        "--context-bins",
        type=int,
        default=30,
        help="Real/precomputed context samples per side.",
    )
    parser.add_argument(
        "--core-len",
        type=int,
        default=800,
        help="Old center window length in samples.",
    )
    parser.add_argument(
        "--event-start-offset",
        type=int,
        default=-80,
        help="Core start offset from EDF task cue, in samples. -80 is -0.5 s at 160 Hz.",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=160.0,
        help="Sample rate for raw context output; non-matching EDF runs are resampled.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap across all requested splits. 0 exports all selected records.",
    )
    parser.add_argument(
        "--drop-edge-padded",
        action="store_true",
        help="Exclude records whose requested context would require edge padding.",
    )
    parser.add_argument("--labels-out", type=Path, default=None)
    parser.add_argument("--subjects-out", type=Path, default=None)
    parser.add_argument("--keys-out", type=Path, default=None)
    args = parser.parse_args()

    if args.context_bins < 0:
        raise ValueError("--context-bins must be non-negative")
    build_context_cache(
        raw_root=args.raw_root,
        processed_dir=args.processed_dir,
        out_path=args.out,
        splits=args.splits,
        context_bins=args.context_bins,
        core_len=args.core_len,
        event_start_offset=args.event_start_offset,
        target_fs=args.target_fs,
        max_records=args.max_records,
        labels_out=args.labels_out,
        subjects_out=args.subjects_out,
        keys_out=args.keys_out,
        drop_edge_padded=args.drop_edge_padded,
    )


if __name__ == "__main__":
    main()
