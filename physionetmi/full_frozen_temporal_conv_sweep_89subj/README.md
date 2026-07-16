# Full Frozen Temporal-Conv Sweep

This sweep used all 89 PhysioNetMI participants with the subject-random split
from the reference run:

`physionetmi/synth_runs/20260714_112036_d128_h256_dep2_bs128_ep150_lr1e-04_lxp0.0_lbt0.0_lcca2.0_sig10.0_s1`

The embedder was frozen for downstream classification. Decoding used hidden
features and the temporal-conv decoder.

## Best Result

Rank 1 is attempt 12:

- Validation accuracy: `0.554089709762533`
- Validation balanced accuracy: `0.5547184988373751`
- Validation macro F1: `0.5508283899186277`
- Test accuracy: `0.5311111111111111`
- Test balanced accuracy: `0.5307865818771456`
- Test macro F1: `0.5351980674882031`

Configuration:

- Frozen embedder: reference run, hidden features
- Decoder: temporal conv
- Decoder hidden dim: `128`
- Decoder depth: `2`
- Decoder kernel size: `15`
- Decoder dropout: `0.4`
- Decoder LR: `3e-4`
- Decoder weight decay: `1e-3`
- Decoder epochs: `50`

## Notes

- The original reference frozen temporal-conv decoder was attempt 1:
  validation accuracy `0.537598944591029`, test accuracy `0.5111111111111111`.
- The best decoder-only baseline before the later sweep was attempt 4:
  validation accuracy `0.5455145118733509`, test accuracy `0.4822222222222222`.
- Wider/newly trained embedding attempts raised embedding zeta in some cases but
  did not improve frozen condition decoding.
- The remaining attempts were reallocated toward decoder hyperparameters around
  the reference embedding after slow long-kernel embedding jobs were clearly
  underperforming.
- Attempt 20 was early-stopped at decoder epoch 25 because its validation
  accuracy was far below the current leader.

Full ranked results are in `results.csv` and `results.json`.
