# PhysioNet condition accuracy sweep, 6-subject screen

This sweep targeted frozen-embedding condition classification accuracy with:

- 50 embedding epochs per run.
- Frozen output embeddings.
- 50 MLP classifier epochs per run.
- 30 total attempts.
- Only temporal filter/kernel combinations with an even split across kernel sizes.

The screen used 6 PhysioNet subjects with 1 held-out test subject because larger 30-subject and 12-subject versions were too slow for this interactive pass. Results are ranked by validation accuracy, with held-out test accuracy reported as a secondary check.

## Top Runs

| rank | attempt | val acc | val bal acc | test acc | d | temporal filters | filters/kernel | kernels | MLP hidden | MLP depth | MLP dropout | MLP lr |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 25 | 0.6889 | 0.6854 | 0.5111 | 128 | 20 | 5 | 15,31,63,127 | 512 | 3 | 0.4 | 0.001 |
| 2 | 29 | 0.6333 | 0.6423 | 0.4667 | 128 | 20 | 5 | 15,31,63,127 | 512 | 2 | 0.4 | 0.001 |
| 3 | 30 | 0.6333 | 0.6285 | 0.4222 | 128 | 20 | 5 | 15,31,63,127 | 512 | 3 | 0.0 | 0.001 |
| 4 | 26 | 0.6111 | 0.6173 | 0.4556 | 128 | 20 | 5 | 15,31,63,127 | 128 | 3 | 0.4 | 0.001 |
| 5 | 27 | 0.6111 | 0.6173 | 0.4333 | 128 | 20 | 5 | 15,31,63,127 | 256 | 3 | 0.4 | 0.001 |

## Main Findings

- The best validation result used the long-kernel bank `15,31,63,127` with `TEMPORAL_FILTERS=20`, giving 5 learned filters per kernel per EEG channel.
- Increasing temporal filters from 16 to 20 on the long-kernel bank produced the largest validation jump: attempt 16 reached 0.5889 val accuracy, while attempt 25 reached 0.6889 with the same decoder shape.
- The best decoder was wide, deep, and regularized: hidden size 512, depth 3, dropout 0.4, learning rate 0.001.
- Dropout mattered. With the best `tf=20` embedding and 512x3 decoder, dropout 0.4 beat dropout 0.0 on validation accuracy: attempt 25 at 0.6889 vs attempt 30 at 0.6333.
- Smaller MLP hidden sizes underperformed the best 512-wide decoder. Attempts 26 and 27 reached 0.6111 with hidden sizes 128 and 256.
- Frozen embedding zeta did not reliably predict condition accuracy. For example, attempt 25 had final frozen zeta 0.0764 but the best validation accuracy, while some higher-zeta runs classified worse.

## Caveats

- This is a 6-subject screen, so validation/test variance is high. Attempt 23 had the best held-out test accuracy, 0.6000, despite ranking lower by validation accuracy.
- The larger 30-subject and 12-subject runs were interrupted for runtime reasons and are not included in this ranking.
- The best candidate should be rerun on a larger subject split before treating it as the final PhysioNet setting.

Full ranked results are in `results.csv`; per-run configs and metrics are in `runs/*/sweep_result.json`.
