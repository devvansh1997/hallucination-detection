"""
20_pilot_token_extraction.py -- Token-Level Extraction Pilot
=============================================================
Evaluates 3 token-extraction strategies on raw (L, T, D) tensors:
  A: Prompt Bottleneck (last prompt token)
  B: Commitment Token  (first generated token)
  C: Max-Energy Token  (max-pool across generated tokens)

Requires raw-token data from 20a_pilot_raw_tokens.py.

Usage:
  python 20_pilot_token_extraction.py --model_folder meta_llama_meta_llama_3_1_8b_instruct
"""

import argparse, gc, os, numpy as np, yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
DATA_DIR = cfg["output"]["data_dir"]
R_L, R_D, RANDOM_SEED = 5, 64, 42

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, required=True)
args = parser.parse_args()


def compute_ul_ud(X_train):
    N, L, D = X_train.shape
    X_f = X_train.permute(1, 0, 2).reshape(L, -1).float()
    A_L = X_f @ X_f.T
    _, U_L = torch.linalg.eigh(A_L.float())
    U_L = torch.flip(U_L[:, -R_L:], dims=[1])
    del X_f, A_L
    X_d = X_train.permute(2, 0, 1).reshape(D, -1)
    A_D = torch.zeros(D, D, dtype=torch.float32)
    for start in range(0, N * L, 50000):
        end = min(start + 50000, N * L)
        A_D.addmm_(X_d[:, start:end].float(), X_d[:, start:end].float().T)
    _, U_D = torch.linalg.eigh(A_D.float())
    U_D = torch.flip(U_D[:, -R_D:], dims=[1])
    return U_L, U_D


def project(X, U_L, U_D):
    temp = torch.matmul(X.float(), U_D)
    G = torch.matmul(temp.transpose(1, 2), U_L).transpose(1, 2)
    return G.reshape(G.shape[0], -1).numpy()


def evaluate_candidate(X, y, prompt_idx, is_known, name):
    """X: (N, 9, 4096), y: (N,). Returns dict of AUROCs."""
    N = X.shape[0]

    # HARP split
    known_prompts = np.where(is_known)[0]
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(known_prompts)
    split = int(len(known_prompts) * 0.75)
    train_p = set(known_prompts[:split])
    valid_p = set(known_prompts[split:])
    unknown_p = np.where(~is_known)[0]
    valid_p.update(unknown_p)

    train_mask = np.array([prompt_idx[i] in train_p for i in range(N)])
    valid_mask = np.array([prompt_idx[i] in valid_p for i in range(N)])
    train_idx = np.where(train_mask)[0]
    valid_idx = np.where(valid_mask)[0]

    if len(train_idx) < 10:
        return None

    # HOSVD
    U_L, U_D = compute_ul_ud(X[train_idx])
    X_tr = project(X[train_idx], U_L, U_D)
    X_va = project(X[valid_idx], U_L, U_D)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    y_tr = y[train_idx]
    y_va = y[valid_idx]

    res = {}
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    res["RF"] = roc_auc_score(y_va, rf.predict_proba(X_va)[:, 1])

    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=RANDOM_SEED)
    lr.fit(X_tr, y_tr)
    res["LR"] = roc_auc_score(y_va, lr.predict_proba(X_va)[:, 1])

    mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                        solver="adam", early_stopping=True,
                        n_iter_no_change=10, max_iter=1000,
                        random_state=RANDOM_SEED)
    mlp.fit(X_tr, y_tr)
    res["MLP"] = roc_auc_score(y_va, mlp.predict_proba(X_va)[:, 1])

    return res


# -- MAIN --
path = os.path.join("../data_unpooled", args.model_folder, "truthfulqa_pilot_raw_tokens.pt")
print(f"Loading: {path}")
data = torch.load(path, weights_only=False)
all_tensors = data["all_tensors"]              # list of lists: per prompt → per beam → (L,T,D)
flags = np.array([int(f) for f in data["all_hallucination_flag"]])
is_known = np.array(data["all_is_known"])
prompt_idx = np.array(data["prompt_indices"])

# Flatten per-prompt beam lists into flat arrays
# Also extract 3 token candidates per beam
all_A, all_B, all_C, all_y, all_pi = [], [], [], [], []

for pi, beam_tensors in enumerate(all_tensors):
    for bi, H in enumerate(beam_tensors):
        H = H[:, 15:24, :].float()             # (9, T, D) — mid layers
        _, T, D = H.shape
        if T == 0:
            zeros = torch.zeros(9, D)
            all_A.append(zeros)
            all_B.append(zeros)
            all_C.append(zeros)
            all_y.append(flags[len(all_y)])
            continue
        # Candidate A
        A = H[:, 0, :]                           # (9, D)
        if A.dim() == 3:
            A = A.squeeze(1)
        B = H[:, 0, :]                           # (9, D) — same
        if B.dim() == 3:
            B = B.squeeze(1)
        C = H.max(dim=1).values                  # (9, D) — max across tokens
        if C.dim() == 3:
            C = C.squeeze(1)

        all_A.append(A)
        all_B.append(B)
        all_C.append(C)
        all_y.append(flags[len(all_y)])
        # prompt_idx already mapped

# Rebuild prompt_idx to match the flattened order
# Since we iterate sequentially, prompt_idx already aligns

print(f"\n  Beams: {len(all_A)}  (layers 15-23)")
print(f"  Truthful: {(np.array(all_y)==0).sum()}  Hallucinated: {(np.array(all_y)==1).sum()}")

results = {}
for name, tensor_list in [("A: Prompt Boundary", all_A),
                           ("B: Commitment Token", all_B),
                           ("C: Max-Energy", all_C)]:
    X = torch.stack(tensor_list)                 # (N, 9, D)
    assert X.ndim == 3, f"Expected 3D, got shape {X.shape}"
    res = evaluate_candidate(X, np.array(all_y), prompt_idx, is_known, name)
    if res:
        results[name] = res
        print(f"  {name:25s}  RF={res['RF']:.4f}  LR={res['LR']:.4f}  MLP={res['MLP']:.4f}")

print(f"\n{'=' * 70}")
print(f"  TOKEN EXTRACTION PILOT RESULTS")
print(f"  {'Candidate':25s}  {'RF':>8s}  {'LR':>8s}  {'MLP':>8s}")
print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")
for name, res in results.items():
    print(f"  {name:25s}  {res['RF']:8.4f}  {res['LR']:8.4f}  {res['MLP']:8.4f}")
