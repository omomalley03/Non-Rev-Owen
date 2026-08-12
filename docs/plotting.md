# Plotting

Core diagnostic plots are still produced by:

```bash
python visualize.py --run mcmaze/runs/<run>
python visualize_synth.py --run physionetmi/synth_runs/<run>
```

Additional report plots:

```bash
python plotting/embedding/plot_embedding_plane_timeseries_fft.py --run mcmaze/runs/<run>
python plotting/mcmaze/plot_mcmaze_conv_kernels.py --run mcmaze/runs/<run>
```

Sweep-summary plots are grouped in `plotting/summary/`.
