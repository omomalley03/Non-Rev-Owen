"""Export PhysioNetMI LMDB records to a ``main_synth.py`` compatible .npy file.

The notebook at ``/home/omo26/data_owen/explore_lmdb.ipynb`` documents the LMDB
layout: ``__keys__`` stores split keys and each record is a pickled dict with
``sample``, ``label``, and ``subject_id``.  ``main_synth.py`` only needs the
sample windows, saved as ``(K, N, T)`` float32.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import numpy.core as _np_core
import numpy.core.multiarray as _np_multiarray
import numpy.core.numeric as _np_numeric


# The provided LMDB was pickled with NumPy 2.x, whose private module path is
# ``numpy._core``.  The training env currently uses NumPy 1.26, so expose the
# equivalent 1.x modules under the 2.x names for unpickling only.
sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
sys.modules.setdefault("numpy._core.numeric", _np_numeric)


def _to_key_bytes(key):
    return key if isinstance(key, bytes) else str(key).encode()


def _parse_splits(value: str) -> list[str]:
    splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("at least one split is required")
    return splits


def export_lmdb(
    processed_dir: Path,
    out_path: Path,
    splits: list[str],
    max_records: int,
    labels_out: Path | None,
    subjects_out: Path | None,
) -> None:
    env = lmdb.open(
        str(processed_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    try:
        with env.begin(write=False) as txn:
            split_keys = pickle.loads(txn.get(b"__keys__"))
            keys = []
            split_counts = {}
            for split in splits:
                if split not in split_keys:
                    available = ", ".join(sorted(split_keys))
                    raise KeyError(f"unknown split {split!r}; available: {available}")
                selected = list(split_keys[split])
                if max_records > 0:
                    remaining = max_records - len(keys)
                    selected = selected[: max(0, remaining)]
                keys.extend(selected)
                split_counts[split] = len(selected)

            if not keys:
                raise ValueError("no records selected")

            first = pickle.loads(txn.get(_to_key_bytes(keys[0])))
            first_sample = np.asarray(first["sample"], dtype=np.float32)
            if first_sample.ndim != 2:
                raise ValueError(f"expected 2D sample windows, got {first_sample.shape}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            windows = np.lib.format.open_memmap(
                out_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(keys),) + first_sample.shape,
            )
            labels = np.empty(len(keys), dtype=np.int64)
            subjects = np.empty(len(keys), dtype=np.int64)

            for idx, key in enumerate(keys):
                record = pickle.loads(txn.get(_to_key_bytes(key)))
                sample = np.asarray(record["sample"], dtype=np.float32)
                if sample.shape != first_sample.shape:
                    raise ValueError(
                        f"record {key!r} has shape {sample.shape}, expected {first_sample.shape}"
                    )
                windows[idx] = sample
                labels[idx] = int(record["label"])
                subjects[idx] = int(record["subject_id"])

            windows.flush()
    finally:
        env.close()

    if labels_out is not None:
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_out, labels)
    if subjects_out is not None:
        subjects_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(subjects_out, subjects)

    sidecar = {
        "source": str(processed_dir),
        "out": str(out_path),
        "splits": splits,
        "split_counts": split_counts,
        "shape": list(windows.shape),
        "dtype": str(windows.dtype),
        "labels_out": str(labels_out) if labels_out is not None else None,
        "subjects_out": str(subjects_out) if subjects_out is not None else None,
    }
    sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps(sidecar, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/home/omo26/data_owen/PhysioNetMI/processed"),
        help="PhysioNetMI processed LMDB directory containing data.mdb.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/physionetmi_train_val.npy"),
        help="Output .npy path for samples with shape (K, N, T).",
    )
    parser.add_argument(
        "--splits",
        type=_parse_splits,
        default=["train", "val"],
        help="Comma-separated LMDB splits to export, e.g. train,val,test.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=int(os.environ.get("PHYSIONETMI_MAX_RECORDS", "0")),
        help="Optional cap across all requested splits. 0 exports all selected records.",
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help="Optional .npy output path for labels.",
    )
    parser.add_argument(
        "--subjects-out",
        type=Path,
        default=None,
        help="Optional .npy output path for subject ids.",
    )
    args = parser.parse_args()

    export_lmdb(
        processed_dir=args.processed_dir,
        out_path=args.out,
        splits=args.splits,
        max_records=args.max_records,
        labels_out=args.labels_out,
        subjects_out=args.subjects_out,
    )


if __name__ == "__main__":
    main()
