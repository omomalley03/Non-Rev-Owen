# Non-Rev

This project trains neural embeddings with non-reversibility and redundancy-control objectives, then evaluates whether those embeddings preserve behaviourally useful information. The main datasets are MC Maze neural activity and Physionet motor-imagery EEG.

The main training entry points are `main.py` for MC Maze and `main_synth.py` for `.npy` window datasets such as Physionet. Core model code is in `model.py`, objective terms are in `loss.py`, the training loop is in `train.py`, data/cache helpers are in `data.py`, `synth_data.py`, and `cache_data.py`, and diagnostic plotting is handled by `visualize.py` and `visualize_synth.py`.

## Prereqs

MC Maze data should be cached with `cache_data.py`.

Physionet should be converted to `.npy` shape `K,N,T`.

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
