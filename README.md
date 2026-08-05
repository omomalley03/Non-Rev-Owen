# Non-Reversibility as a Pretraining Objective for Spatio-Temporal Foundation Models

This project trains neural embeddings with non-reversibility and redundancy-control objectives, then evaluates whether those embeddings preserve behaviourally useful information. The main datasets are MC Maze neural activity and Physionet motor-imagery EEG.

The main training entry points are `main.py` for MC Maze and `main_synth.py` for `.npy` window datasets such as Physionet. Core model code is in `model.py`, objective terms are in `loss.py`, the training loop is in `train.py`, data/cache helpers are in `data.py`, `synth_data.py`, and `cache_data.py`, and diagnostic plotting is handled by `visualize.py` and `visualize_synth.py`.

## Prereqs

MC Maze data should be cached with `cache_data.py`.

Physionet should be converted to `.npy` shape `K,N,T`. The current
`physionetmi_config.sh` uses an all-splits padded cache: the center 800 samples
are copied exactly from the old LMDB-derived cache, 30 samples of raw EDF
context are added on each side, and end-of-run trials without full real context
are removed. The default Physionet training split is `SYNTH_SPLIT=subject_random`
over the exported subjects.

To regenerate the padded Physionet cache after unpacking the PhysioNet ZIP into
`physionetmi_raw/files`:

```bash
micromamba run -n nonrev python prepare_physionetmi_context.py \
  --out cache/physionetmi_all_context30_noedge.npy \
  --labels-out cache/physionetmi_all_context30_noedge_labels.npy \
  --subjects-out cache/physionetmi_all_context30_noedge_subjects.npy \
  --keys-out cache/physionetmi_all_context30_noedge_keys.txt \
  --splits train,val,test \
  --drop-edge-padded
```

## Pre-training Embeddings

### MC Maze

```bash
source mcmaze_config.sh
python main.py
```

`mcmaze_config.sh` has comments explaining model/training hyperparams.

### Physionet

```bash
source physionetmi_config.sh
python main_synth.py
```

## Training Decoder

```bash
python predict_mcmaze_velocity.py
python predict_physionet_condition.py
```
