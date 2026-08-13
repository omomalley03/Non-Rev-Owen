# Non-Reversibility as a Pretraining Objective for Learning Foundation Models of Spatio-Temporal Time Series

This repository trains maximally non-reversible embeddings of neural data (encoding), then evaluates whether those embeddings preserve behaviourally useful information (decoding). The main report datasets are MC Maze neural activity and PhysioNet Motor Movement/Imagery EEG. Corresponding author: Owen O'Malley (omo26@cam.ac.uk [until October 2027] / owen.m.omalley@gmail.com [forever]).

The core training entry points remain at the repository root:

- `main.py`: MC Maze embedding pretraining.
- `main_synth.py`: For data saved as `.npy`, used for PhysioNetMI and synthetic data which are saved as shape (K,N,T) arrays.
- `visualize.py` and `visualize_synth.py`: standard plots.

Supporting scripts are grouped by role:

- `configs/`: shell configs for datasets and environments.
- `data_prep/`: cache and dataset conversion scripts.
- `decoders/`: downstream MC Maze velocity and PhysioNet condition decoders.
- `plotting/`: report and diagnostic plotting scripts.
- `sweep_scripts/`: MC Maze and PhysioNet sweep launchers -- the statistical analysis experiments in the dissertation
- `docs/`: workflow notes and repo map.

See `docs/repo_map.md`, `docs/workflows.md`, `docs/sweeps.md`, and `docs/plotting.md` for the expanded guide.

## Setup

Install the project dependencies in your Python environment. `requirements.txt` records the Python packages used by the scripts. 


## MC Maze

MC Maze data can be cached before repeated runs to speed up loading:

```bash
python data_prep/cache_data.py
```

Train an embedding and run the standard decoder/visualization workflow:

```bash
source configs/mcmaze_config.sh
python main.py
python decoders/predict_mcmaze_velocity.py --run mcmaze/runs/<run>
python visualize.py --run mcmaze/runs/<run>
```

## PhysioNetMI

PhysioNetMI is converted to `.npy` windows with shape `K,N,T`. The current config uses the all-splits padded cache described in `configs/physionetmi_config.sh`.

To regenerate the padded PhysioNetMI cache after unpacking the PhysioNet ZIP into `physionetmi_raw/files`:

```bash
micromamba run -n nonrev python data_prep/prepare_physionetmi_context.py \
  --out cache/physionetmi_all_context30_noedge.npy \
  --labels-out cache/physionetmi_all_context30_noedge_labels.npy \
  --subjects-out cache/physionetmi_all_context30_noedge_subjects.npy \
  --keys-out cache/physionetmi_all_context30_noedge_keys.txt \
  --splits train,val,test \
  --drop-edge-padded
```

Train an embedding and run the standard decoder/visualization workflow:

```bash
source configs/physionetmi_config.sh
python main_synth.py
python decoders/predict_physionet_condition.py --run physionetmi/synth_runs/<run>
python visualize_synth.py --run physionetmi/synth_runs/<run>
```

The default PhysioNet training split is `SYNTH_SPLIT=subject_random` over exported subjects. To hold out test subjects while mixing the remaining participants between train and validation trials:

```bash
export SYNTH_SPLIT="subject_random"
export SYNTH_HOLDOUT_SUBJECT_COUNT="10"
export VAL_SPLIT="0.2"
```

To use distinct train, validation, and test participants:

```bash
export SYNTH_SPLIT="subject_holdout"
export SYNTH_VAL_SUBJECT_COUNT="19"
export SYNTH_HOLDOUT_SUBJECT_COUNT="20"
```

In `subject_holdout`, all loaded trials from selected validation subjects go to validation, and all loaded trials from held-out subjects go to the test-only split.

## Common Plots

```bash
python plotting/embedding/plot_embedding_plane_timeseries_fft.py --run mcmaze/runs/<run>
python plotting/mcmaze/plot_mcmaze_conv_kernels.py --run mcmaze/runs/<run>
python plotting/physionet/visualize_physionet_participant_split.py --run physionetmi/synth_runs/<run>
```
