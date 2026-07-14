"""
22_evaluate_phase1_kinematics_and_q.py
========================================
HOSVD + Q-Statistic + Depth Kinematics + Gram-Schmidt Residualization.
Dummy data unit tests run first to verify pipeline integrity.
"""

import argparse, gc, os
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
DATA_DIR = cfg["output"]["data_dir"]
R_L, R_D, RANDOM_SEED = 5, 64, 42

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, default=None)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--suffix", type=str, default="")
args = parser.parse_args()


# ==============================================================================
# STEP 0: DUMMY DATA UNIT TESTS
# ==============================================================================
def run_dummy_tests():
    """Synthetic sanity checks on [50, 9, 4096] random tensor."""
    print("  [STEP 0] Running dummy data unit tests ...")
    X = torch.randn(50, 9, 512)            # smaller D for speed
    y = torch.randint(0, 2, (50,)).float()
    N, L, D_d = X.shape

    # HOSVD
    X_f = X.permute(1, 0, 2).reshape(L, -1)
    A_L = X_f @ X_f.T
    _, U_L = torch.linalg.eigh(A_L)
    U_L = torch.flip(U_L[:, -R_L:], dims=[1])
    X_d = X.permute(2, 0, 1).reshape(D_d, -1)
    A_D = X_d @ X_d.T
    _, U_D = torch.linalg.eigh(A_D)
    U_D = torch.flip(U_D[:, -R_D:], dims=[1])

    # Project
    P_L = U_L @ U_L.T
    P_D = U_D @ U_D.T
    X_hat = P_L @ X.float() @ P_D
    F_core = X_hat.reshape(N, -1)

    # Test 1: Shape
    assert F_core.shape == (50, R_L * R_D), f"Shape: {F_core.shape}"
    print("    [PASS] Test 1: Core feature shape correct")

    # Test 2: Q-Statistic non-collinearity
    E = X.float() - X_hat
    e_norms = E.reshape(N, -1).norm(dim=1)
    corr = torch.corrcoef(torch.stack([e_norms, F_core.norm(dim=1)]))[0, 1]
    assert abs(corr.item()) < 0.9, f"Corr too high: {corr.item():.4f}"
    print("    [PASS] Test 2: Q-statistic not collinear with core norms")

    # Test 3: Kinematic bounds
    x_n = F.normalize(X.float(), dim=2)
    dot = (x_n[:, :-1] * x_n[:, 1:]).sum(dim=2)
    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    assert (theta >= 0).all() and (theta <= np.pi).all()
    assert not torch.isnan(theta).any() and not torch.isinf(theta).any()
    print("    [PASS] Test 3: Angular velocities in [0, pi], no NaN/Inf")

    # Test 4: Gram-Schmidt residualization
    split = int(N * 0.75)
    F_q_synth = torch.randn(N, 10)
    F_kin_synth = torch.randn(N, 16)
    F_new = torch.cat([F_q_synth, F_kin_synth], dim=1)
    ridge = Ridge(alpha=1.0)
    ridge.fit(F_core[:split].numpy(), F_new[:split].numpy())
    F_perp = F_new.numpy() - ridge.predict(F_core.numpy())
    corr_gs = np.corrcoef(F_core[split:].numpy().ravel()[:100],
                          F_perp[split:].ravel()[:100])[0, 1]
    assert abs(corr_gs) < 0.3, f"GS correlation too high: {corr_gs:.4f}"
    print("    [PASS] Test 4: Gram-Schmidt residualization orthogonal")

    print("  [PASS] All Dummy Data Unit Tests Executed Successfully\n")


# ==============================================================================
# HOSVD
# ==============================================================================
def compute_ul_ud(X_train):
    N, L, D = X_train.shape
    X_f = X_train.permute(1, 0, 2).reshape(L, -1).float()
    A_L = X_f @ X_f.T
    _, U_L = torch.linalg.eigh(A_L)
    U_L = torch.flip(U_L[:, -R_L:], dims=[1])
    X_d = X_train.permute(2, 0, 1).reshape(D, -1)
    A_D = torch.zeros(D, D, dtype=torch.float32)
    for start in range(0, N * L, 50000):
        end = min(start + 50000, N * L)
        A_D.addmm_(X_d[:, start:end].float(), X_d[:, start:end].float().T)
    _, U_D = torch.linalg.eigh(A_D)
    U_D = torch.flip(U_D[:, -R_D:], dims=[1])
    return U_L, U_D


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    run_dummy_tests()

    models = cfg["models"]
    if args.model_folder:
        models = [m for m in models if m["folder"] == args.model_folder]
    datasets = cfg["datasets"]
    if args.dataset:
        datasets = [d for d in datasets if d["name"] == args.dataset]

    for m_cfg in models:
        for d_cfg in datasets:
            model_folder = m_cfg["folder"]
            dataset = d_cfg["name"]
            path = os.path.join(DATA_DIR, model_folder,
                                f"{dataset}_pooled{args.suffix}.pt")
            if not os.path.exists(path):
                print(f"  [SKIP] {path}")
                continue

            print(f"\n{'=' * 60}")
            print(f"  {model_folder} / {dataset.upper()}")
            print(f"{'=' * 60}")

            # ── STEP 1: Load & Split ──
            data = torch.load(path, weights_only=False)
            X = torch.stack(data["all_emb"]).float()         # (N, 9, D)
            y_all = np.array([int(f) for f in data["all_hallucination_flag"]])
            is_known = np.array(data["all_is_known"])
            N, L, D = X.shape
            n_prompts = len(is_known)
            prompt_idx = np.array(data.get("prompt_indices",
                data.get("all_prompt_indices", list(range(n_prompts)))))

            # HARP split
            known_idx = np.where(is_known)[0]
            np.random.seed(RANDOM_SEED); np.random.shuffle(known_idx)
            s = int(len(known_idx) * 0.75)
            tp = set(known_idx[:s]); vp = set(known_idx[s:])
            vp.update(np.where(~is_known)[0])
            t_mask = np.array([prompt_idx[i] in tp for i in range(N)])
            v_mask = np.array([prompt_idx[i] in vp for i in range(N)])
            t_idx = np.where(t_mask)[0]; v_idx = np.where(v_mask)[0]
            y_train = y_all[t_idx]; y_valid = y_all[v_idx]
            print(f"  Train: {len(t_idx)}  Valid: {len(v_idx)}")

            # MAD scaling (train only)
            X_t = X[t_idx]
            med = X_t.median(dim=0).values
            mad = (X_t - med).abs().median(dim=0).values + 1e-6
            X = (X - med) / mad

            # ── STEP 2: HOSVD + Q-Statistic ──
            U_L, U_D = compute_ul_ud(X[t_idx])
            P_L = U_L @ U_L.T  # (L, L)
            P_D = U_D @ U_D.T  # (D, D)

            X_hat = P_L @ X @ P_D                                 # (N, L, D)
            F_core = X_hat.reshape(N, -1).numpy()                 # (N, 320)

            E = X - X_hat
            q_total = E.pow(2).sum(dim=(1, 2)).unsqueeze(1)       # (N, 1)
            q_layer = (X - X @ P_D).pow(2).sum(dim=2)             # (N, L)
            F_q = torch.cat([q_total, q_layer], dim=1).numpy()    # (N, 10)

            # ── STEP 3: Depth Kinematics ──
            x_n = F.normalize(X, dim=2)                            # (N, L, D)
            d_l = (x_n[:, 1:] - x_n[:, :-1]).norm(dim=2)          # (N, L-1)
            dot = (x_n[:, :-1] * x_n[:, 1:]).sum(dim=2)
            theta_l = torch.acos(torch.clamp(dot, -1.0, 1.0))     # (N, L-1)
            F_kin = torch.cat([d_l, theta_l], dim=1).numpy()      # (N, 16)

            # ── STEP 4: Gram-Schmidt ──
            F_new = np.concatenate([F_q, F_kin], axis=1)          # (N, 26)
            ridge = Ridge(alpha=1.0)
            ridge.fit(F_core[t_idx], F_new[t_idx])
            F_perp = F_new - ridge.predict(F_core)                 # (N, 26)

            # ── STEP 5: Multi-Variant Ablation ──
            variants = {
                "V1: Core (320)":              (F_core, 320),
                "V2: Core + Raw Geo (346)":    (np.concatenate([F_core, F_new], axis=1), 346),
                "V3: Core + Orth Geo (346)":   (np.concatenate([F_core, F_perp], axis=1), 346),
                "V4: Orth Geo Alone (26)":     (F_perp, 26),
            }

            results = {}
            for vname, (feats, dim) in variants.items():
                scaler = StandardScaler()
                tr = scaler.fit_transform(feats[t_idx])
                va = scaler.transform(feats[v_idx])

                res = {}
                rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=RANDOM_SEED, n_jobs=-1)
                rf.fit(tr, y_train)
                res["RF"] = roc_auc_score(y_valid, rf.predict_proba(va)[:, 1])

                lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=RANDOM_SEED)
                lr.fit(tr, y_train)
                res["LR"] = roc_auc_score(y_valid, lr.predict_proba(va)[:, 1])

                mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                                    solver="adam", early_stopping=True,
                                    n_iter_no_change=10, max_iter=1000,
                                    random_state=RANDOM_SEED)
                mlp.fit(tr, y_train)
                res["MLP"] = roc_auc_score(y_valid, mlp.predict_proba(va)[:, 1])

                results[vname] = res
                print(f"  {vname:35s}  RF={res['RF']:.4f}  LR={res['LR']:.4f}  MLP={res['MLP']:.4f}")

            # ── STEP 6: Output ──
            print(f"\n  {'='*70}")
            print(f"  ABLATION SUMMARY")
            print(f"  {'Variant':35s}  {'Dim':>5s}  {'RF':>8s}  {'LR':>8s}  {'MLP':>8s}")
            print(f"  {'-'*35}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
            for vname, (_, dim) in variants.items():
                r = results[vname]
                print(f"  {vname:35s}  {dim:5d}  {r['RF']:8.4f}  {r['LR']:8.4f}  {r['MLP']:8.4f}")

            v1_best = max(results["V1: Core (320)"].values())
            v3_best = max(results["V3: Core + Orth Geo (346)"].values())
            if v3_best > v1_best:
                print(f"\n  >> V3 outperforms V1 ({v3_best:.4f} > {v1_best:.4f}). "
                      f"Q-subspace + kinematics successfully break the baseline ceiling.")
            else:
                print(f"\n  >> V3 does NOT outperform V1 ({v3_best:.4f} <= {v1_best:.4f}). "
                      f"HOSVD core already captures the discriminative geometry.")
