# Plotting

Figure-generation scripts are grouped by purpose.

- `mcmaze/`: MC Maze embedding, plane, velocity, and kernel plots.
- `physionet/`: PhysioNet participant-split visualizations.
- `embedding/`: dataset-agnostic embedding diagnostics such as time-series FFTs.
- `summary/`: sweep summary and report-level scatter/bar plots.
- `misc/`: standalone visualizations.

Most scripts can be run directly from the repository root, for example:

```bash
python plotting/embedding/plot_embedding_plane_timeseries_fft.py --run mcmaze/runs/<run>
python plotting/mcmaze/plot_mcmaze_conv_kernels.py --run mcmaze/runs/<run>
```
