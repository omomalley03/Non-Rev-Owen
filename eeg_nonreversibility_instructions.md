# Claude Code Instructions: EEG Non-Reversibility MLP (MCMaze)

## Project Goal

Build a full training pipeline for an MLP `f_θ` that embeds EEG windows from the **MCMaze dataset** by maximising a non-reversibility pretraining objective `S(F_k)`.

---

## Objective

The loss to **maximise** is:

$$S(F_k) = \frac{2}{K^2} \sum_{k,k'} \left[ \operatorname{Tr}^2\!\left(F_k F_{k'}^\top\right) - \operatorname{Tr}\!\left(\left(F_k F_{k'}^\top\right)^2\right) \right]$$

Where `F_k = f_θ(X_k) ∈ ℝ^(d × T)` is the MLP embedding of window `k`. Use `-S` as the training loss.

---

## Step 1 — Environment & Data

### Dependencies
Install:
```
torch, numpy, scipy, nlb_tools, einops, tqdm, matplotlib
```

### Loading MCMaze
- Load the MCMaze dataset using `nlb_tools`.
- **Before writing any windowing code**, inspect the dataset interactively:
  - Number of neural/EEG channels `C`
  - Sampling rate (Hz)
  - Trial structure and lengths (are trials variable length?)
- Use this inspection to decide on window size `T` and stride. If trials are short relative to a sensible fixed window, use **trial-aligned epochs** rather than sliding windows, and note this in a comment.

### Preprocessing: Gaussian Smoothing
Apply Gaussian smoothing to each channel independently before windowing:

```python
from scipy.ndimage import gaussian_filter1d

# sigma is in samples; default = round(0.020 * sampling_rate)
X_smooth = gaussian_filter1d(X_raw, sigma=sigma, axis=-1)
```

`sigma` should be a configurable hyperparameter (default ≈ 20 ms in samples).

### Windowing
Segment smoothed data into windows `X_k` of shape `(C, T)`. Hold out 10% of windows as a validation split.

---

## Step 2 — MLP Model (`model.py`)

The model embeds **each timepoint independently** using a shared-weight MLP:
`x_t : (C,) → f_t : (d,)`, applied across all T timesteps with the same weights.

**Architecture:**
1. Input is a single snapshot `(C,)` — no flattening of time
2. Three fully-connected layers with **LayerNorm + GELU** activations
3. Output is `(d,)`

Apply across time by reshaping: `(B, C, T)` → `(B*T, C)`, pass through MLP, reshape back to `(B, T, d)`, then transpose to `(B, d, T)` to give `F_k`.

**Configurable hyperparameters:** embedding dim `d` (default 128), hidden dim, depth (default 3 layers).

**Weight initialisation:** orthogonal initialisation on all linear layers.

```python
class MLP(nn.Module):
    def __init__(self, in_channels, d=128, hidden_dim=256, depth=3):
        ...
    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        x = x.permute(0, 2, 1).reshape(B * T, C)  # (B*T, C)
        x = self.net(x)                             # (B*T, d)
        x = x.reshape(B, T, d).permute(0, 2, 1)    # (B, d, T)
        return x
        # Same MLP weights applied to every snapshot — time-invariant
```

---

## Step 3 — Loss Function (`loss.py`)

Given a batch of `K` windows embedded as `F ∈ ℝ^(K, d, T)`:

### Efficient einsum implementation

```python
# Tr(F_k @ F_k'.T) for all pairs (k, k')
# F_k @ F_k'.T has shape (d, d), its trace = sum of elementwise products
# einsum over d and T dimensions:
trace_1 = torch.einsum('kit,kjt->ij', F, F)   # shape: (K, K)
# ^^^ This computes Tr(F_k F_k'.T) for all pairs

# Tr((F_k @ F_k'.T)^2):
# = Tr(G G) where G = F_k F_k'.T
# = sum_{a,b} G_{ab} G_{ba} = sum_{a,b} G_{ab}^2 (since G is symmetric here)
# Efficiently via einsum:
trace_2 = torch.einsum('kim,kjm,kin,kjn->ij', F, F, F, F)  # shape: (K, K)
# ^^^ This contracts over d (i,j indices) and T (m,n indices)

S = (2 / K**2) * (trace_1**2 - trace_2).sum()
loss = -S   # maximise S
```

Add a docstring explaining the geometric interpretation: **S measures how asymmetrically (non-reversibly) the embedding trajectories evolve across window pairs — maximising it encourages the MLP to capture temporal irreversibility in the neural data.**

---

## Step 4 — Config (`config.py` or `config.yaml`)

All hyperparameters in one place:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_ms` | 20 | Gaussian smoothing std (ms) |
| `window_size` | TBD | Timesteps per window (set after dataset inspection) |
| `stride` | TBD | Sliding window stride |
| `d` | 128 | Embedding dimension (per-snapshot) |
| `hidden_dim` | 256 | MLP hidden layer width |
| `depth` | 3 | Number of MLP layers |
| `batch_size` | 64 | Windows per batch |
| `epochs` | 50 | Training epochs |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `val_split` | 0.1 | Fraction of windows held out |

---

## Step 5 — Training Pipeline (`train.py`)

- **DataLoader** over windowed dataset, shuffled, with configurable batch size.
- **Optimiser:** AdamW with `lr` and `weight_decay` from config.
- **LR Scheduler:** `CosineAnnealingWarmRestarts`.
- **Training loop:**
  - Log loss per step and per epoch.
  - After each epoch, evaluate `S` on the validation split.
  - Track best validation `S`; save checkpoint to `checkpoints/best.pt`.
- **At end of training:** save loss curve to `outputs/loss_curve.png`.

---

## Step 6 — Entry Point (`main.py`)

`python main.py` should run the full pipeline:

1. Load & preprocess data
2. Window & split
3. Instantiate model
4. Train
5. Print summary:
   - Final train `S`
   - Best validation `S`
   - Total training time

---

## Notes for Claude Code

- **Inspect the dataset first** before writing windowing logic — do not assume trial lengths or sampling rate.
- **If trials are variable length**, use trial-aligned epochs and pad/truncate to a fixed `T` with a comment explaining the choice.
- **Comment every einsum** with what contraction is being computed and the resulting shape.
- The two einsum expressions for `trace_2` can be memory-intensive for large `K` and `d` — consider chunking over pairs if needed and note this as a potential optimisation.
