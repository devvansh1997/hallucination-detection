# Population HOSVD Hallucination Pipeline

End-to-end mechanistic interpretability pipeline for detecting hallucination signatures in LLM residual-stream activations via tensor decomposition.

## Anatomy

| Constant | Value | Meaning |
|----------|-------|---------|
| L | 32 | Transformer layers |
| D | 4096 | Hidden dimension |
| N | 100 | Mock training population |
| R_L | 5 | Layer-mode rank |
| R_D | 64 | Hidden-mode rank |

## Pipeline (5 Stages)

1. **Synthetic Harvest** — Generate mock residual-stream tensors `(32, T_i, 4096)` with variable sequence lengths and binary truth/lie labels.
2. **Token Neutralization** — Mean-pool across the token axis to collapse each sample to `(32, 4096)`. Stack into `X_population` of shape `(100, 32, 4096)`.
3. **Gram Matrix Trick** — Unfold along layer and hidden modes, form Gram matrices, eigendecompose to extract orthonormal factor matrices `U_L (32 x 5)` and `U_D (4096 x 64)`.
4. **Core Compression & Centroid Vault** — Project each sample via `G_i = U_L^T @ x_i @ U_D` into core tensors `(5, 64)`. Compute mean centroids `C_truth` and `C_lie`.
5. **Test-Time Detective** — For an unseen activation, project through the training factor matrices and compute Frobenius distance to each centroid.

## Files

```
data.py    — SyntheticRaw Dataset class (generation + token pooling)
utils.py   — gram_factor_matrices, get_g_pop, project_to_core
main.py    — Full pipeline orchestration
```

## Run

```bash
python main.py
```

### Expected Output

```
{'sub_dC': tensor(5.7293), 'sub_dH': tensor(5.7541), 'sub_diff': tensor(-0.0248), 'core_norm': tensor(5.2125)}
```

- `sub_dC` — Frobenius distance to truth centroid
- `sub_dH` — Frobenius distance to hallucination centroid
- `sub_diff` — `sub_dC - sub_dH` (negative = closer to truth)
- `core_norm` — Frobenius norm of the compressed test core

## Dependencies

- `torch`
