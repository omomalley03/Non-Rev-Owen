# Sweep Scripts

Sweep launchers are split by dataset.

- `mcmaze/`: MC Maze training and fine-tuning sweeps.
- `physionet/`: PhysioNetMI training and condition-decoder sweeps.
- `common/`: shared aggregation, fine-tuning, and reporting helpers.

Run sweep scripts from the repository root so relative output paths stay under `mcmaze/` or `physionetmi/`.

```bash
python -u sweep_scripts/mcmaze/run_mcmaze_dim_train_finetune_sweep.py
python -u sweep_scripts/physionet/run_physionet_dim_train_finetune_sweep.py
```
