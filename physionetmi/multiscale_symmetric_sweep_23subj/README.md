# PhysioNetMI Multiscale Symmetric Temporal Filter Sweep

Screening sweep completed with 20 successful 30-epoch attempts.

Scope:
- Dataset/cache: `cache/physionetmi_train_val.npy`
- Screening split: 23 randomly selected PhysioNetMI subjects, seed 1, 20% validation
- Model base: `D=128`, `HIDDEN_DIM=256`, `DROPOUT=0.2`, `LAMBDA_BLOCK_CCA=1.0`
- Front-end: `TEMPORAL_FRONTEND=multiscale_symmetric`
- Ranking metric: best validation `zeta = C- / C+`
- Full ranking: `results.csv`

## Top Results

| Rank | Attempt | zeta | S | C+ | Filters | Kernels | Conv layers | MLP depth | LR |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 10 | 0.3195 | 0.2963 | 0.9219 | 16 | `3,7,15,31,63` | 2 | 2 | 1e-3 |
| 2 | 6 | 0.3189 | 0.4517 | 1.4098 | 16 | `7,15,31,61` | 2 | 2 | 1e-3 |
| 3 | 14 | 0.3169 | 0.3994 | 1.2521 | 16 | `5,11,23,47,95` | 2 | 2 | 1e-3 |
| 4 | 9 | 0.3164 | 0.3190 | 1.0049 | 16 | `15,31,63,127` | 2 | 2 | 1e-3 |
| 5 | 4 | 0.3157 | 0.2800 | 0.8806 | 8 | `7,15,31,61` | 2 | 2 | 1e-3 |
| 6 | 11 | 0.3153 | 0.2313 | 0.7279 | 8 | `3,7,15,31,63` | 2 | 2 | 1e-3 |

## Readout

The clear signal is that two multiscale symmetric conv layers are much stronger than one layer. Every top-seven run used two temporal conv layers except none; one-layer variants clustered around zeta 0.25-0.27.

The best pure-zeta setting was:

```sh
export TEMPORAL_FRONTEND="multiscale_symmetric"
export TEMPORAL_FILTERS="16"
export MULTISCALE_SYMMETRIC_CONV_LAYERS="2"
export RESIDUAL_KERNELS="3,7,15,31,63"
export DEPTH="2"
export LR="1e-3"
export EPOCHS="30"
```

If we care about avoiding the highest C+ inflation, the best balanced candidate is attempt 11:

```sh
export TEMPORAL_FILTERS="8"
export MULTISCALE_SYMMETRIC_CONV_LAYERS="2"
export RESIDUAL_KERNELS="3,7,15,31,63"
export DEPTH="2"
export LR="1e-3"
```

It was only about 0.004 zeta below the winner, while using lower C+ and half the temporal filters.

Lower learning rates did not look promising in this 30-epoch screen. The `1e-4` runs underfit badly, and `3e-4` reduced C+ but lost too much zeta. For the next full-size run, use `LR=1e-3` first and rely on checkpoint selection by zeta.

## Artifacts

- `results.csv`: ranked table for all 20 successful attempts
- `results.json`: full structured result records
- `runs/*/outputs/log.csv`: per-epoch train/validation loss components
- `runs/*/checkpoints/best.pt`: best-zeta checkpoint for each candidate
- `logs/attempt_*.log`: raw training output per attempt

Note: `runs/05_tf4_k3-7-15-31-63_cl1_dep2_lr0.001` is a partial failed setup
from before candidate validation was added; `TEMPORAL_FILTERS=4` cannot be split
across five kernel branches. It is not included in `results.csv` or
`results.json`.
