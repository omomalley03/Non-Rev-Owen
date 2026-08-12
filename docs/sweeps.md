# Sweeps

MC Maze sweep launchers live in `sweep_scripts/mcmaze/`; PhysioNetMI sweep launchers live in `sweep_scripts/physionet/`.

Commands to run sweeps in Chapter 4 / Chapter 5 of the thesis:

```bash
python -u sweep_scripts/mcmaze/run_mcmaze_dim_train_finetune_sweep.py
python -u sweep_scripts/mcmaze/run_mcmaze_even_even_vs_50_50_sweep.py
python -u sweep_scripts/mcmaze/run_mcmaze_odd_plane_fraction_finetune_sweep.py

python -u sweep_scripts/physionet/run_physionet_dim_train_finetune_sweep.py
python -u sweep_scripts/physionet/run_physionet_even_even_vs_50_50_sweep.py
python -u sweep_scripts/physionet/run_physionet_odd_plane_fraction_finetune_sweep.py
```

Shared aggregation and fine-tuning helpers are in `sweep_scripts/common/`.

Raw paper sweep outputs and logs are stored under:

```text
sweep_results/hyperparam_sweep_paper_logs/
```
which includes per seed results, confidence intervals, and paired t-test results.