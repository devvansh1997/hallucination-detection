# Population HOSVD Hallucination Pipeline — Real Data Report

## 1. Pipeline Adaptation: Synthetic → Real Residual Stream

The pipeline originally developed on mock Gaussian tensors (`SyntheticRaw`) was ported to
real LLM residual-stream activations from `res2_train_32_layers_tensor.pt`.

### Structural changes

| Property | Synthetic Data | Real Data |
|----------|---------------|-----------|
| Organization | Sample-first list | Layer-first dict of lists |
| Anatomy detection | Hardcoded constants | Inferred from checkpoint |
| Sample count | N=100 | N=3,979 |
| Sequence lengths | Uniform [4, 16] | Variable, real tokenizations |
| Label balance | Exact 50/50 | 58.8% truthful / 41.2% hallucinated |

### Code changes

1. **`RawActivations` Dataset** — loads `.pt` checkpoint, sorts layer keys numerically,
   transposes layer-first → sample-first. L, D, N inferred from data.
2. **`pool_activations()`** — generic mean-pool + stack, decoupled from dataset class.
3. **Dynamic `gram_factor_matrices()`** — reshape dims derived from `X_population.shape`.
4. **Eigenvector ordering fix** — `torch.flip` applied after `eigh` so Mode 0 corresponds
   to the largest eigenvalue (descending variance order).

---

## 2. Full Dataset Distribution (3,979 Samples)

All samples projected through full-population factor matrices, analyzed by label group.
Eigenvectors are now correctly ordered descending by eigenvalue.

### Population summary

| Group | Count | Percentage |
|-------|-------|-----------|
| Truthful (label=0) | 2,341 | 58.8% |
| Hallucinated (label=1) | 1,638 | 41.2% |

### Core Frobenius norms

| Statistic | Truthful | Hallucinated |
|-----------|----------|-------------|
| Mean | 377.45 | 393.18 |
| Std | 215.62 | 235.58 |
| Min | 74.90 | 74.11 |
| Max | 1,281.51 | 1,287.96 |
| Median | 325.61 | 315.44 |
| Centroid norm | 364.38 | 381.26 |

**Key observation:** The mean difference is 15.73, but the standard deviation is
~215–235 — the distributions overlap almost completely. The centroid distance
of 19.49 is dwarfed by the within-class variance.

### Layer-mode energy (correctly ordered, descending)

| Mode | Truthful | Hallucinated |
|------|----------|-------------|
| 0 (dominant) | **20,236.9** | **17,929.3** |
| 1 | 5,203.5 | 4,305.9 |
| 2 | 2,123.5 | 1,798.9 |
| 3 | 874.0 | 740.0 |
| 4 | 502.0 | 423.7 |

Mode 0 captures the overwhelming majority of residual-stream variance. Energy
decays by ~4× per mode — consistent with the low-rank structure of transformer
representations.

### Cross-group separation

| Metric | Value |
|--------|-------|
| Centroid distance (Frobenius) | 19.49 |
| Mode 0 diff | **17.94** (92% of total) |
| Mode 1 diff | 6.81 |
| Mode 2 diff | 3.17 |
| Mode 3 diff | 1.10 |
| Mode 4 diff | 0.62 |

Separation is concentrated almost entirely in Mode 0.

---

## 3. 80/20 Train/Test Split — AUROC Evaluation

A strict 80/20 random split was performed (seed=42). Factor matrices and centroids
were computed **exclusively** from the training set (3,183 samples). The remaining
796 samples were scored using `sub_diff = ||G_test - C_truth|| - ||G_test - C_lie||`.

### Results

| Metric | Value |
|--------|-------|
| Test samples | 796 |
| Hallucination rate | 40.3% |
| Accuracy (sub_diff > 0 → hallucination) | 50.63% |
| **AUROC** | **0.508** |

### Interpretation

An AUROC of 0.508 is statistically indistinguishable from random guessing (0.500).
The `sub_diff` feature, operating on mean-pooled token representations compressed
via population HOSVD, **does not separate truthful from hallucinated samples** in
this dataset.

---

## 4. Why This Approach Failed

### 4.1 Token pooling dilutes the hallucination signal (critical)

A hallucination typically occurs at a specific token or span (e.g., *"The capital
of France is London"* — one corrupted token among many factual ones). Mean-pooling
across all tokens in the sequence averages the hallucinatory token with 10+ factual
tokens, washing out the signal. The centroid distance of 19.49 is almost certainly
dominated by prompt-level variance (topic, length, style) rather than hallucination
signatures.

**Remediation path:** Replace mean-pooling with position-aware methods:
- Extract only the final token embedding (where generated content lives)
- Use attention-weighted aggregation focused on hallucination-prone positions
- Apply contrastive token selection (subtract prompt tokens from completion tokens)

### 4.2 Within-class variance overwhelms between-class separation

The Frobenius norm standard deviation (~220) is 14× larger than the mean difference
between classes (~16). The two distributions are not distinct clusters — they are a
single overlapping cloud. A linear decision boundary (centroid distance) cannot
separate them.

**Remediation path:** Instead of raw centroid distance, feed the full `(5, 64)` core
tensor as a flattened feature vector into a non-linear classifier (Random Forest,
XGBoost, or a small MLP) that can learn a decision boundary within the overlapping
region.

### 4.3 HOSVD basis may not align with the discriminative subspace

The HOSVD factor matrices `U_L` and `U_D` are computed to maximize **reconstruction**
of the population variance, not **discrimination** between classes. The dominant modes
capture prompt-level structure (topic, syntax) that is orthogonal to hallucination.
A discriminative decomposition (e.g., Multilinear Discriminant Analysis) would be
more appropriate.

---

## 5. Summary

| Claim | Status |
|-------|--------|
| Pipeline runs on real data | Confirmed |
| HOSVD captures low-rank structure | Confirmed |
| Mean-pooled core centroids differ measurably | Confirmed (19.49) |
| Core distance separates truth from hallucination | **Refuted** (AUROC = 0.508) |

The infrastructure is sound, but the `sub_diff` metric on mean-pooled cores is not
a viable hallucination detector. The next iteration must address token-level signal
localization and discriminative subspace learning.
