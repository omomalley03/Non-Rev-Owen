# Repo Map

Core training code remains at the repository root:

- `main.py`: MC Maze pretraining entry point.
- `main_synth.py`: `.npy` window pretraining entry point used for PhysioNetMI and FACED.
- `config.py`, `paths.py`: configuration and output-path helpers.
- `model.py`, `loss.py`, `train.py`: model, objectives, and training loop.
- `data.py`, `synth_data.py`: MC Maze and `.npy` dataset loading.
- `visualize.py`, `visualize_synth.py`, `visualize_loss.py`: training diagnostics.

Supporting scripts are grouped into `configs/`, `data_prep/`, `decoders/`, `plotting/`, `sweep_scripts/`, and `experiments/`.

Generated outputs, checkpoints, caches, raw datasets, and logs are intentionally ignored by Git.
