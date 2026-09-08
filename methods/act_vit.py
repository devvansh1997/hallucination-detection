"""methods/act_vit.py -- ACT-ViT (Bar-Shalom et al., NeurIPS 2025) under OUR split and metric.

THE ARCHITECTURE IS THEIRS, IMPORTED, NOT REIMPLEMENTED. `ACT_Vit_foundation` comes straight from
../ACT-ViT/utils/Architectures.py, as does the training recipe: AdamW, BCELoss, cosine schedule with
10% warmup, batch 128. If their model changes, ours changes with it.

WHY WE DO NOT RUN THEIR DRIVER. `utils/dataset_preprocess.py:40` splits with StratifiedKFold on
labels -- flat over responses, no grouping. That is correct for them, because they generate ONE
response per query, so a flat split is a question split. Our data is 10 beams per question, so the
same call puts a question's answers on both sides. That is precisely our answer-level protocol, the
leaky one this paper is about. Running `foundation_main.py` unmodified would produce a number under
the leaky split and label it ACT-ViT. So: their model, our harness, identical treatment to HARP.

ORIENTATION, WHICH IS THE EASIEST THING HERE TO GET BACKWARDS. Their label convention is y=1 for a
CORRECT response, so their sigmoid output is P(correct) and high means truthful. Every score in this
harness runs the other way: high = predicted hallucination. Rather than train their way and negate
-- a second place to make a sign error -- we train directly on `data.labels` (1 = hallucinated), so
the output is P(hallucinated) and needs no flip. BCE is symmetric under flipping both label and
output, so this is the same model, not a modified one. The self-test asserts the direction.

INPUT. The pooled activation tensors from 59_extract_act_tensors.py, whose `act_pool` is asserted
bit-identical to their own preprocessing. Their per-LLM adapters are sized at
max(FEATURE_DIMS) = 4096, so Qwen's 3584 is zero-padded to 4096 -- their design, matched here.

CONFIGURATION. Their published single-dataset defaults, fixed, no grid search (deliberate: a 24-cell
grid x 5 seeds x 2 protocols is not affordable, and a grid selected on anything touching test would
be worse than no grid). num_layers 3, hidden_dim 128, heads 4, dropout 0.3, patch (1,1), lr 1e-3,
weight_decay 1e-3, 15 epochs, batch 128. Early stopping on a validation split carved from the
TRAINING rows only, grouped by question, at their TRAIN_VAL_RATIO of 4/5.
"""

import os
import sys

import numpy as np

from .base import Method

HERE = os.path.dirname(os.path.abspath(__file__))
ACT_REPO = os.path.abspath(os.path.join(HERE, "..", "..", "ACT-ViT"))
DEFAULT_AT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "data-acttensors"))

# utils/constants.py FEATURE_DIMS, in order -- the index selects which adapter runs.
LLM_INDEX = {"qwen-2.5-7b-instruct": 2, "llama-3.1-8b": 1, "llama-3.1-8b-instruct": 1}
FEATURE_DIM_MAX = 4096

EPOCHS = 15
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 5
VAL_FRACTION = 1.0 / 5.0        # their TRAIN_VAL_RATIO = 4/5


class _Args:
    """The attribute bag their get_model() reads. Values are their published defaults."""
    probe_model = "ACT-Vit-foundation"
    hidden_dim = 128
    heads = 4
    dropout = 0.3
    num_layers = 3
    patch_size = "(1,1)"
    pool = "cls"


def build_model(l_pool, n_pool, device):
    """Import and instantiate THEIR architecture. Kept in one place so the import path is auditable."""
    if not os.path.isdir(ACT_REPO):
        raise SystemExit(
            "ACT-ViT not found at %s. Clone it next to HARP-Code:\n"
            "    git clone https://github.com/BarSGuy/ACT-ViT %s" % (ACT_REPO, ACT_REPO))
    if ACT_REPO not in sys.path:
        sys.path.insert(0, ACT_REPO)
    try:
        from utils.Architectures import get_model
    except ImportError as e:
        raise SystemExit(
            "could not import ACT-ViT's architecture (%s). Install with --no-deps:\n"
            "    pip install --no-deps vit-pytorch==1.8.9 einops==0.8.0\n"
            "--no-deps IS REQUIRED. A plain install pulls torchvision, which pip builds\n"
            "against a different torch than the env has; transformers imports torchvision.io\n"
            "unconditionally, so every model load in the env then dies with\n"
            "'operator torchvision::nms does not exist'. vit-pytorch's ViT never uses it." % e)
    return get_model(_Args(), input_shape=(l_pool, n_pool, FEATURE_DIM_MAX),
                     input_dim=FEATURE_DIM_MAX).to(device)


def pad_features(batch, target=FEATURE_DIM_MAX):
    """Zero-pad the feature axis to their adapters' input width. (B, L, N, D) -> (B, L, N, target)."""
    import torch
    d = batch.shape[-1]
    if d == target:
        return batch
    if d > target:
        raise SystemExit("hidden size %d exceeds ACT-ViT's adapter width %d" % (d, target))
    return torch.nn.functional.pad(batch, (0, target - d))


def grouped_val_split(prompt_id, train_idx, seed, frac=VAL_FRACTION):
    """Carve a validation set out of TRAINING rows, whole questions at a time.

    Grouped, not stratified-flat: a validation set that shares questions with training would report
    an optimistic AUC and early-stop at the wrong epoch. Never touches test_idx."""
    q = np.unique(prompt_id[train_idx])
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(q)[:max(1, int(round(len(q) * frac)))].tolist())
    mask = np.array([int(p) in held for p in prompt_id[train_idx]])
    if mask.all() or not mask.any():          # degenerate on tiny inputs; fall back to no val
        return train_idx, None
    return train_idx[~mask], train_idx[mask]


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def train_and_score(at, labels, prompt_id, tr, te, l_pool, n_pool, device, seed,
                    epochs=EPOCHS, log=None):
    """Their recipe. Returns P(hallucinated) for rows `te`, in that order."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import get_scheduler

    torch.manual_seed(seed)
    np.random.seed(seed)

    fit_idx, val_idx = grouped_val_split(prompt_id, tr, seed)
    model = build_model(l_pool, n_pool, device)
    llm_idx = torch.zeros(BATCH, dtype=torch.long, device=device)   # resized per batch below

    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(idx.astype(np.int64))),
                          batch_size=BATCH, shuffle=shuffle)

    def forward(rows):
        x = torch.from_numpy(np.asarray(at[rows], dtype=np.float32)).to(device)
        x = pad_features(x)
        return model(x, llm_idx[:len(rows)]).reshape(-1)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = max(1, int(np.ceil(len(fit_idx) / BATCH))) * epochs
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=int(0.1 * steps),
                          num_training_steps=steps)
    criterion = torch.nn.BCELoss()

    best, best_state, stale = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for (b,) in loader(fit_idx, True):
            rows = b.numpy()
            opt.zero_grad()
            # labels are 1 = HALLUCINATED, so the output is P(hallucinated). See module docstring.
            loss = criterion(forward(rows),
                             torch.from_numpy(labels[rows].astype(np.float32)).to(device))
            loss.backward()
            opt.step()
            sched.step()

        if val_idx is None:
            continue
        model.eval()
        with torch.no_grad():
            vs = np.concatenate([forward(b.numpy()).cpu().numpy()
                                 for (b,) in loader(val_idx, False)])
        a = _auc(labels[val_idx], vs)
        if log and (ep == 0 or ep == epochs - 1):
            log("      epoch %2d/%d  val AUROC %.4f" % (ep + 1, epochs, a))
        if np.isnan(a) or a <= best:
            stale += 1
            if stale >= PATIENCE:
                break
        else:
            best, stale = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return np.concatenate([forward(b.numpy()).cpu().numpy()
                               for (b,) in loader(te, False)])


class ActViT(Method):
    name = "act_vit"
    granularity = "beam"
    description = "ACT-ViT (Bar-Shalom et al. 2025), their architecture, our split and metric"

    def add_args(self, parser):
        parser.add_argument("--at-dir", default=DEFAULT_AT_DIR,
                            help="where 59_extract_act_tensors.py wrote its .npz files")
        parser.add_argument("--n-eff", type=int, default=20,
                            help="token pooling of the extraction to use (20 or 100)")
        parser.add_argument("--l-eff", type=int, default=8)
        parser.add_argument("--epochs", type=int, default=EPOCHS)

    def configure(self, args):
        self.at_dir = args.at_dir
        self.n_eff = args.n_eff
        self.l_eff = args.l_eff
        self.epochs = args.epochs

    def precompute(self, data):
        path = os.path.join(getattr(self, "at_dir", DEFAULT_AT_DIR), data.model_folder,
                            "%s_at_L%d_N%d.npz" % (data.dataset, getattr(self, "l_eff", 8),
                                                   getattr(self, "n_eff", 20)))
        if not os.path.exists(path):
            raise SystemExit(
                "%s not found. Extract first:\n"
                "    python 59_extract_act_tensors.py --dataset %s --model_folder %s\n"
                "then derive this N_eff on CPU:\n"
                "    python 59_extract_act_tensors.py --repool-from <the N100 file> --n-pool %d"
                % (path, data.dataset, data.model_folder, getattr(self, "n_eff", 20)))

        z = np.load(path)
        at, beam_row = z["at"], z["beam_row"]

        # Align the extraction's row order to the harness's. 59 writes in question-sorted order,
        # which is NOT the order the bundle holds, and a silent misalignment here would train on
        # correctly-shaped nonsense and produce a plausible AUROC.
        order = np.empty(len(beam_row), dtype=np.int64)
        order[beam_row] = np.arange(len(beam_row))
        if len(beam_row) != len(data.labels):
            raise SystemExit("extraction has %d rows, the pinned data has %d -- was it run with "
                             "--limit?" % (len(beam_row), len(data.labels)))
        if not np.array_equal(z["label"][order], data.labels):
            raise SystemExit("labels in %s disagree with the pinned generations; the extraction is "
                             "stale, re-run 59" % path)
        if not np.array_equal(z["prompt_id"][order], data.prompt_id):
            raise SystemExit("prompt_id in %s disagrees with the pinned generations" % path)

        self._diag = {"at_path": path, "l_eff": int(at.shape[1]), "n_eff": int(at.shape[2]),
                      "hidden": int(at.shape[3]), "gb": round(at.nbytes / 1024 ** 3, 2),
                      "median_completion_tokens": int(np.median(z["token_len"]))}
        print("    [act_vit] %s  %s  %.1f GB" % (os.path.basename(path), at.shape,
                                                 at.nbytes / 1024 ** 3), flush=True)
        return {"at": at, "order": order}

    def score(self, data, pre, train_idx, test_idx):
        at, order = pre["at"], pre["order"]
        # at is in extraction order; index it through `order` so row i of the bundle maps correctly.
        view = at[order]
        return train_and_score(view, data.labels, data.prompt_id, train_idx, test_idx,
                               view.shape[1], view.shape[2], data.device,
                               seed=int(train_idx[0]) if len(train_idx) else 0,
                               epochs=getattr(self, "epochs", EPOCHS), log=print)

    def meta(self):
        return {"orientation": "trained on label 1 = hallucinated, so output is P(hallucinated); "
                               "high = predicted hallucination, no sign flip applied",
                "architecture": "ACT_Vit_foundation imported from ../ACT-ViT, unmodified",
                "config": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                           "weight_decay": WEIGHT_DECAY, "hidden_dim": _Args.hidden_dim,
                           "num_layers": _Args.num_layers, "heads": _Args.heads,
                           "patch_size": _Args.patch_size, "dropout": _Args.dropout,
                           "grid_search": False},
                **getattr(self, "_diag", {})}

    def self_test(self):
        import torch

        # Feature padding to their adapter width, which is max(FEATURE_DIMS) not our hidden size.
        x = torch.zeros(2, 8, 20, 3584)
        assert pad_features(x).shape == (2, 8, 20, 4096)
        assert torch.equal(pad_features(x)[..., 3584:], torch.zeros(2, 8, 20, 512))
        print("    [PASS] Qwen's 3584 zero-pads to ACT-ViT's 4096 adapter width")

        # The validation carve must not share questions with the rows it is selected against.
        pid = np.repeat(np.arange(20), 10)
        tr = np.arange(200)
        fit, val = grouped_val_split(pid, tr, seed=0)
        assert val is not None and len(fit) + len(val) == len(tr)
        assert not (set(pid[fit].tolist()) & set(pid[val].tolist())), \
            "validation questions leaked into the fitting rows"
        print("    [PASS] validation split is grouped by question (%d fit / %d val rows, no shared "
              "questions)" % (len(fit), len(val)))

        # Row alignment. beam_row is a permutation; inverting it wrongly is a silent failure, so
        # pin that the inverse maps extraction order back to bundle order.
        beam_row = np.array([2, 0, 3, 1])
        order = np.empty(4, dtype=np.int64)
        order[beam_row] = np.arange(4)
        payload = np.array([20, 0, 30, 10])          # value at extraction row i is 10 * beam_row[i]
        assert list(payload[order]) == [0, 10, 20, 30], list(payload[order])
        print("    [PASS] beam_row inversion restores bundle order")

        if not os.path.isdir(ACT_REPO):
            print("    [SKIP] ../ACT-ViT not cloned -- model build and orientation not exercised")
            return

        # End to end on separable synthetic data. This is the orientation test: hallucinated rows
        # carry a large positive signal, so if the score is oriented correctly AUROC goes to 1, and
        # if it is inverted it goes to 0. Getting this backwards inverts every reported cell, which
        # has already happened once on this project (EigenScore).
        rng = np.random.default_rng(0)
        n, L, N, D = 64, 8, 4, 16
        y = np.array([0, 1] * (n // 2))
        pid = np.repeat(np.arange(n // 2), 2)
        at = rng.standard_normal((n, L, N, D)).astype(np.float32) * 0.01
        at[y == 1] += 3.0
        tr, te = np.arange(0, 48), np.arange(48, n)
        s = train_and_score(at, y, pid, tr, te, L, N, "cpu", seed=0, epochs=4)
        a = _auc(y[te], s)
        assert a > 0.9, ("orientation looks INVERTED or the model did not learn: AUROC %.3f. "
                         "High score must mean hallucinated." % a)
        print("    [PASS] end to end on separable data: AUROC %.3f, high score = hallucinated" % a)
