import torch


def pool_activations(activation_list):
    """Mean-pool tokens (dim=1) for each tensor in a list, then stack.
    Works on any subset — training or test.
    """
    x_pooled_list = [x_i.mean(dim=1) for x_i in activation_list]
    return torch.stack(x_pooled_list)


def gram_factor_matrices(X_population, R_L, R_D):
    # calculate: U_L
    # unfold X_pop along the layer
    X_permuted_l = torch.permute(X_population, [1, 0, 2])  # [32, 100, 4096]
    N_pop, L_pop, D_pop = X_population.shape
    X_unfolded_l = torch.reshape(
        X_permuted_l, (L_pop, N_pop * D_pop)
    )

    # create gram matrix
    A_L = X_unfolded_l @ torch.transpose(X_unfolded_l, 0, 1)  # [L_pop, L_pop]

    # eigh doesn't support bfloat16 on CPU — cast to float32 for the solve,
    # then cast eigenvectors back to the original dtype
    _A_L = A_L.to(torch.float32)
    _, eigvecs_l = torch.linalg.eigh(_A_L)
    eigvecs_l = eigvecs_l.to(A_L.dtype)
    # eigh returns ascending; flip so Mode 0 = largest eigenvalue
    U_L = torch.flip(eigvecs_l[:, -R_L:], dims=[1])  # [L_pop, R_L]

    # calculate: U_D
    # unfold X_pop along the depth
    X_permuted_d = torch.permute(X_population, [2, 0, 1])  # [4096, 100, 32]
    X_unfolded_d = torch.reshape(
        X_permuted_d, (D_pop, N_pop * L_pop)
    )

    # create gram matrix
    A_D = X_unfolded_d @ torch.transpose(X_unfolded_d, 0, 1)  # [D_pop, D_pop]

    _A_D = A_D.to(torch.float32)
    _, eigvecs_d = torch.linalg.eigh(_A_D)
    eigvecs_d = eigvecs_d.to(A_D.dtype)
    U_D = torch.flip(eigvecs_d[:, -R_D:], dims=[1])  # [D_pop, R_D]

    return U_L, U_D


def get_g_pop(X_population, U_L, U_D):
    G_population_list = []
    # loop over each sample
    for i in range(len(X_population)):
        G_i = torch.transpose(U_L, 0, 1) @ X_population[i] @ U_D
        G_population_list.append(G_i)

    # create G_Pop tensor
    G_population = torch.stack(G_population_list)
    return G_population


def analyze_core_distributions(checkpoint_path, R_L=5, R_D=64):
    """Load full dataset, project all samples to core space, and print
    detailed distribution statistics for truth vs hallucination groups."""
    from data import RawActivations

    dataset = RawActivations(checkpoint_path)
    X_pop = pool_activations(dataset.raw_activation_list)
    U_L, U_D = gram_factor_matrices(X_pop, R_L, R_D)
    G_pop = get_g_pop(X_pop, U_L, U_D)  # (N, 5, 64)
    y = dataset.y_train

    G_truth = G_pop[y == 0]
    G_lie = G_pop[y == 1]

    def _report(name, cores):
        if cores.shape[0] == 0:
            print(f"\n  {name}: NO SAMPLES")
            return None
        norms = torch.norm(cores.reshape(cores.shape[0], -1), dim=1)
        flat = cores.flatten()
        centroid = cores.mean(dim=0)
        layer_nrg = torch.norm(cores, dim=(0, 2))
        print(f"\n  {'='*54}")
        print(f"   {name}")
        print(f"  {'='*54}")
        print(f"   Count:               {cores.shape[0]}")
        print(f"   Core shape:          {tuple(cores.shape[1:])}")
        print(f"   Frobenius norms:")
        print(f"     mean ± std:        {norms.mean():.4f} ± {norms.std():.4f}")
        print(f"     min / max:         {norms.min():.4f} / {norms.max():.4f}")
        print(f"     median:            {norms.median():.4f}")
        print(f"   Entry values (all):")
        print(f"     mean ± std:        {flat.mean():.6f} ± {flat.std():.6f}")
        print(f"     min / max:         {flat.min():.6f} / {flat.max():.6f}")
        print(f"   Centroid norm:       {torch.norm(centroid):.4f}")
        print(f"   Layer-mode energy (||G[:,k,:]||):")
        for k in range(cores.shape[1]):
            print(f"     mode {k}: {layer_nrg[k]:.4f}")
        return centroid

    print(f"\n  Population: {dataset.N} samples  |  L={dataset.L}  D={dataset.D}")
    print(f"  Truthful: {G_truth.shape[0]}  |  Hallucinated: {G_lie.shape[0]}")

    C_t = _report("TRUTHFUL (label=0)", G_truth)
    C_h = _report("HALLUCINATED (label=1)", G_lie)

    if C_t is not None and C_h is not None:
        print(f"\n  {'='*54}")
        print(f"   CROSS-GROUP")
        print(f"  {'='*54}")
        print(f"   Centroid distance:   {torch.norm(C_t - C_h):.4f}")
        for k in range(C_t.shape[0]):
            print(f"     mode {k} diff: {torch.norm(C_t[k] - C_h[k]):.4f}")


def _compute_auroc(scores, labels):
    """Manual AUROC via trapezoidal rule.
    scores: higher → more likely class 1.  labels: 0/1 tensor."""
    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order].float()
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = torch.cat([torch.zeros(1), torch.cumsum(labels_sorted, dim=0) / n_pos])
    fpr = torch.cat([torch.zeros(1), torch.cumsum(1 - labels_sorted, dim=0) / n_neg])
    return torch.trapz(tpr, fpr).item()


def evaluate_auroc(checkpoint_path, R_L=5, R_D=64, test_frac=0.2, seed=42):
    """80/20 train/test split. Train centroids on 80%, score all 20%,
    compute AUROC of sub_diff as a hallucination detector."""
    from data import RawActivations

    dataset = RawActivations(checkpoint_path)
    N = dataset.N
    y_all = dataset.y_train

    torch.manual_seed(seed)
    indices = torch.randperm(N)
    split_idx = int(N * (1 - test_frac))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    # -- Train --
    train_list = [dataset.raw_activation_list[i] for i in train_idx]
    y_train_sub = y_all[train_idx]
    X_train = pool_activations(train_list)
    U_L, U_D = gram_factor_matrices(X_train, R_L, R_D)
    G_train = get_g_pop(X_train, U_L, U_D)
    C_truth, C_lie = project_to_core(G_train, y_train_sub)

    # -- Test --
    sub_diffs = []
    y_test = y_all[test_idx]
    for idx in test_idx:
        x_raw = dataset.raw_activation_list[idx]
        x_pool = x_raw.mean(dim=1)            # (32, 4096)
        G_test = U_L.T @ x_pool @ U_D          # (5, 64)
        sub_dC = torch.norm(G_test - C_truth)
        sub_dH = torch.norm(G_test - C_lie)
        sub_diffs.append((sub_dC - sub_dH).item())

    sub_diffs = torch.tensor(sub_diffs)
    auroc = _compute_auroc(sub_diffs, y_test)

    # Simple accuracy at threshold 0
    preds = (sub_diffs > 0).long()
    acc = (preds == y_test).float().mean().item()

    n_train = len(train_idx)
    n_test = len(test_idx)
    n_halluc = (y_test == 1).sum().item()

    print(f"\n  Train samples: {n_train}  |  Test samples: {n_test}")
    print(f"  Test hallucination rate: {n_halluc}/{n_test} "
          f"({n_halluc/n_test*100:.1f}%)")
    print(f"  Accuracy (sub_diff > 0 → hallucination): {acc:.4f}")
    print(f"  AUROC: {auroc:.4f}")

    return auroc, sub_diffs, y_test


def project_to_core(G_population, y_train):
    # compute truths vs lies
    C_truth = G_population[y_train == 0].mean(dim=0)
    C_lie = G_population[y_train == 1].mean(dim=0)

    return C_truth, C_lie


def evaluate_classifiers(checkpoint_path, R_L=5, R_D=64,
                         test_size=0.2, random_state=42):
    """Strict no-leakage pipeline:
    1. Load dataset & split indices FIRST.
    2. Compute U_L, U_D from TRAINING activations only.
    3. Project BOTH train and test through training-only factor matrices.
    4. Flatten cores → 2D features, train classifiers, report AUROC."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from data import RawActivations

    dataset = RawActivations(checkpoint_path)
    y_all = dataset.y_train.numpy().astype(np.int64)
    N = dataset.N

    # Split indices FIRST — before any factor matrix computation
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y_all, random_state=random_state
    )

    # -- Train: compute U_L, U_D from training pool only --
    train_list = [dataset.raw_activation_list[i] for i in train_idx]
    X_train_pooled = pool_activations(train_list)
    U_L, U_D = gram_factor_matrices(X_train_pooled, R_L, R_D)

    # -- Project training cores --
    G_train = get_g_pop(X_train_pooled, U_L, U_D)

    # -- Project test cores through SAME training-only factor matrices --
    test_pooled_list = [dataset.raw_activation_list[i].mean(dim=1) for i in test_idx]
    X_test_pooled = torch.stack(test_pooled_list)
    G_test = get_g_pop(X_test_pooled, U_L, U_D)

    # -- Flatten and classify --
    X_train_flat = G_train.reshape(G_train.shape[0], -1).float().numpy()
    X_test_flat  = G_test.reshape(G_test.shape[0], -1).float().numpy()
    y_train_np   = y_all[train_idx]
    y_test_np    = y_all[test_idx]

    results = {}

    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=random_state)
    lr.fit(X_train_flat, y_train_np)
    lr_auc = roc_auc_score(y_test_np, lr.predict_proba(X_test_flat)[:, 1])
    results["LogisticRegression"] = lr_auc

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=random_state, n_jobs=-1)
    rf.fit(X_train_flat, y_train_np)
    rf_auc = roc_auc_score(y_test_np, rf.predict_proba(X_test_flat)[:, 1])
    results["RandomForest"] = rf_auc

    n_train, n_test = len(train_idx), len(test_idx)
    n_halluc = y_test_np.sum()
    print(f"\n  Feature dim: {X_train_flat.shape[1]}  "
          f"(flattened {tuple(G_train.shape[1:])})")
    print(f"  Train: {n_train}  |  Test: {n_test}")
    print(f"  Test hallucination rate: {n_halluc}/{n_test} "
          f"({n_halluc/n_test*100:.1f}%)")
    print(f"  {'─'*42}")
    for name, auc in results.items():
        print(f"  {name:25s}  AUROC = {auc:.4f}")

    return results
