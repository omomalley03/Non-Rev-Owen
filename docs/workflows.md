# Workflows

## MC Maze

```bash
source configs/mcmaze_config.sh
python main.py
python decoders/predict_mcmaze_velocity.py --run mcmaze/runs/<run>
python visualize.py --run mcmaze/runs/<run>
```

## PhysioNetMI

```bash
source configs/physionetmi_config.sh
python main_synth.py
python decoders/predict_physionet_condition.py --run physionetmi/synth_runs/<run>
python visualize_synth.py --run physionetmi/synth_runs/<run>
```

## PhysioNetMI Cache

```bash
python data_prep/prepare_physionetmi_context.py \
  --out cache/physionetmi_all_context30_noedge.npy \
  --labels-out cache/physionetmi_all_context30_noedge_labels.npy \
  --subjects-out cache/physionetmi_all_context30_noedge_subjects.npy \
  --keys-out cache/physionetmi_all_context30_noedge_keys.txt \
  --splits train,val,test \
  --drop-edge-padded
```
