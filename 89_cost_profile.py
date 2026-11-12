"""
89_cost_profile.py -- measured time and peak memory, stage by stage, for LOTUS, flat PCA and ACT-ViT.
=====================================================================================================
THE QUESTIONS (ICLR 2027 reviews).
  zjPm: "Reduced projection size and representation storage do not directly establish lower runtime or
        peak memory. A matched comparison of activation extraction, projection fitting, and detector
        inference would help substantiate the efficiency claim and distinguish offline costs from
        per-response overhead."
  pjMe: "Could the authors clarify the computational cost of learning the cross-model feature mapping?"

STAGES (each runs in its own process, so its peak memory is its own):
  extract      GPU. The LLM forward pass over the first --questions questions of the pinned answers,
               then each detector's pooling of that answer: LOTUS's summaries (57.pool_answer on the
               nine window layers, the route 65 checks against the pinned features) and ACT-ViT's
               (59.act_pool over every layer, L8 N20). Per-answer milliseconds, peak GPU memory,
               stored bytes per answer. The forward pass is shared; the pooling is what differs.
  fit_lotus    CPU. LOTUS on one question-level split (seed 42): scaling fit, Tucker-2 basis fit,
               projection, random-forest fit (OFFLINE), then scoring the test answers (PER ANSWER).
               Bit-identical features to 61's hosvd arm.
  fit_flatpca  CPU. The same with flat PCA at the same width (61's flat_pca arm).
  act_vit      GPU. ACT-ViT on the same split: loading its tensors, one training run (their recipe,
               15 epochs), and per-answer inference of the same architecture.
  mapping      CPU. Cross-model transfer: the ridge map from the target's cores to the source's
               (66.fit_alignment, penalty chosen by grouped CV) and applying it per answer, plus the
               source model's reading of the target's training answers (65, from its meta file).

  python 89_cost_profile.py --self-test
  python 89_cost_profile.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct --stages fit_lotus fit_flatpca
  python 89_cost_profile.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct --stages extract act_vit
  python 89_cost_profile.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct --stages mapping \
      --source llama-3.1-8b
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEFAULT_OUT = os.path.join(HERE, "results", "rebuttal")
CONDITION = "triple_concat"
STAGES = ("extract", "extract_gpu", "fit_lotus", "fit_flatpca", "act_vit", "mapping")
WINDOW = list(range(16, 25))          # hidden-state indices of blocks 15..23 (65.WINDOW)
SEED = 42


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_mods = {}


def mods():
    if not _mods:
        _mods["s61"] = _load("s61", "61_flatten_control.py")
        _mods.update(_mods["s61"].mods())            # s43, s26, s44
        _mods["s02"] = _load("s02", "28_eval_band.py")
    return _mods


def peak_rss_gb():
    """Peak resident memory of THIS process in GB (Linux ru_maxrss is in KB). None if unavailable."""
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2, 3)
    except ImportError:
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 1024.0 ** 3, 3)
        except Exception:
            return None


class Clock:
    def __init__(self, sync=None):
        self.t, self.sync = {}, sync

    def __call__(self, name):
        clock = self

        class _C:
            def __enter__(self_):
                if clock.sync:
                    clock.sync()
                self_.t0 = time.perf_counter()

            def __exit__(self_, *a):
                if clock.sync:
                    clock.sync()
                clock.t[name] = clock.t.get(name, 0.0) + time.perf_counter() - self_.t0
        return _C()


def stored_floats_per_answer(D):
    """Numbers stored per answer, as float16 bytes = 2x these."""
    return {"lotus": 43 * D, "act_vit_L8_N20": 8 * 20 * D, "act_vit_L8_N5": 8 * 5 * D,
            "act_vit_L4_N10": 4 * 10 * D}


# ---------------------------------------------------------------------------------------------
# CPU: LOTUS and flat PCA, with fitting and per-answer scoring timed separately
# ---------------------------------------------------------------------------------------------

def time_reduction(kind, feats, spec, tr, te, y, seed, readout="RF"):
    """kind in {"lotus", "flat_pca"}. Returns (timings dict, feature matrix for all rows)."""
    m = mods()
    s02, s43, s26, s61 = m["s02"], m["s43"], m["s26"], m["s61"]
    ck = Clock()
    Zs, apply_fns, n_params, macs = [], [], 0, 0
    for i, (key, r_l, r_d) in enumerate(spec):
        X = feats[key]
        N, L, F = X.shape
        Xf = X.reshape(N, L * F)
        with ck("offline_scaling_fit"):
            params = s02.fit_robust_scale(Xf[tr])
        with ck("offline_scaling_apply_train"):
            Xs = s02.apply_robust_scale(Xf, params).reshape(N, L, F)
        if kind == "lotus":
            with ck("offline_basis_fit"):
                U_L, U_D = s43.compute_ul_ud_randomized(Xs[tr], r_l, r_d, seed + i)
            with ck("offline_project_train"):
                Z = s26.project_core(Xs, U_L, U_D)
            n_params += L * r_l + F * r_d
            macs += L * F * r_d + L * r_l * r_d
            apply_fns.append(lambda Xt, p=params, a=U_L, b=U_D, L=L, F=F: s26.project_core(
                s02.apply_robust_scale(Xt.reshape(len(Xt), L * F), p).reshape(len(Xt), L, F), a, b))
        else:
            M = Xs.reshape(N, -1)
            with ck("offline_basis_fit"):
                mu, comp = s61._fit_pca(M[tr], r_l * r_d, seed + i)
            with ck("offline_project_train"):
                Z = (M - mu) @ comp
            n_params += L * F * r_l * r_d + L * F
            macs += L * F * r_l * r_d
            apply_fns.append(lambda Xt, p=params, mu=mu, cp=comp, L=L, F=F: (
                s02.apply_robust_scale(Xt.reshape(len(Xt), L * F), p) - mu) @ cp)
        Zs.append(Z)
    Zall = np.concatenate(Zs, axis=1)

    from sklearn.ensemble import RandomForestClassifier
    with ck("offline_classifier_fit"):
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed,
                                     n_jobs=-1).fit(Zall[tr], y[tr])
    # Per answer: raw summaries in, score out -- scaling, projection and the forest, on test rows only.
    raw = {key: feats[key][te] for key, _, _ in spec}
    with ck("per_answer_total"):
        Zt = np.concatenate([fn(raw[key]) for fn, (key, _, _) in zip(apply_fns, spec)], axis=1)
        s = clf.predict_proba(Zt)[:, 1]
    with ck("per_answer_classifier_only"):
        clf.predict_proba(Zt)
    from sklearn.metrics import roc_auc_score
    out = {k: round(v, 4) for k, v in ck.t.items()}
    out["offline_total_seconds"] = round(sum(v for k, v in ck.t.items() if k.startswith("offline")), 3)
    out["per_answer_ms"] = round(1000.0 * ck.t["per_answer_total"] / len(te), 4)
    out["per_answer_ms_without_classifier"] = round(
        1000.0 * (ck.t["per_answer_total"] - ck.t["per_answer_classifier_only"]) / len(te), 4)
    out.update({"fitted_basis_parameters": int(n_params), "projection_macs_per_answer": int(macs),
                "width": int(Zall.shape[1]), "n_train": int(len(tr)), "n_test": int(len(te)),
                "test_pooled_auroc": float(roc_auc_score(y[te], s))})
    return out, Zall


def stage_fit(kind, dataset, model_folder, data_dir):
    m = mods()
    import methods.base as B
    c = B.canonical()
    t = time.time()
    feats, y, pid, is_known = m["s44"].load_new_dataset(dataset, data_dir, model_folder,
                                                         condition=CONDITION)
    load_s = time.time() - t
    y = np.asarray(y, dtype=int)
    tr, te = (np.asarray(v, dtype=int) for v in c["question_split"](is_known, np.asarray(pid), len(y), SEED))
    res, _ = time_reduction(kind, feats, m["s44"].CONDITION_SPECS[CONDITION], tr, te, y, SEED)
    res["load_seconds"] = round(load_s, 1)
    res["stored_gb_loaded"] = round(sum(v.nbytes for v in feats.values()) / 1024 ** 3, 3)
    return res


# ---------------------------------------------------------------------------------------------
# GPU: extraction and ACT-ViT
# ---------------------------------------------------------------------------------------------

def lotus_pool_torch(h, q_hi=0.95, q_lo=0.05):
    """57.pool_answer on a torch tensor h (L1, T, D), for timing the pooling on the GPU. Same maths:
    max over tokens, the 0.95 / 0.05 token quantiles (linear interpolation, numpy's default), and the
    same quantiles of the layer-to-layer update h[i+1] - h[i]."""
    import torch
    q = torch.tensor([q_hi, q_lo], dtype=h.dtype, device=h.device)
    qs = torch.quantile(h, q, dim=1)
    d = h[1:] - h[:-1]
    qd = torch.quantile(d, q, dim=1)
    return {"core": h.amax(dim=1), "q95": qs[0], "q05": qs[1], "v95": qd[0], "v05": qd[1]}


def act_pool_torch(A, l_pool=8, n_pool=20, n_max=100):
    """59.act_pool on a torch tensor A (L, N, D): zero-pad to (L_for_pad, N_for_pad), then max over
    non-overlapping blocks. The replicate pad in their code never fires at these settings (asserted)."""
    import torch
    L, N, D = A.shape
    l_for_pad = (int(L / l_pool) + 1) * l_pool
    n_for_pad = n_max if (n_max % n_pool) == 0 else (int(n_max / n_pool) + 1) * n_pool
    P = torch.zeros((l_for_pad, n_for_pad, D), dtype=A.dtype, device=A.device)
    P[:min(L, l_for_pad), :min(N, n_for_pad)] = A[:min(L, l_for_pad), :min(N, n_for_pad)]
    assert l_for_pad % l_pool == 0 and n_for_pad % n_pool == 0
    return P.reshape(l_pool, l_for_pad // l_pool, n_pool, n_for_pad // n_pool, D).amax(dim=(1, 3))


def stage_extract(dataset, model_folder, data_dir, device, questions, gpu_pool=False):
    import torch
    import yaml
    from transformers import AutoModelForCausalLM
    s57 = _load("s57", "57_extract_all_layers.py")
    s59 = _load("s59", "59_extract_act_tensors.py")
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    model_id = next(mm["id"] for mm in cfg["models"] if mm["folder"] == model_folder)
    seq = torch.load(os.path.join(data_dir, model_folder, "%s_sequences_v1.pt" % dataset),
                     weights_only=False)
    input_ids, prompt_lens = seq["input_ids"], seq["prompt_len"]
    pid = np.asarray(seq["prompt_id"])
    order = np.argsort(pid, kind="stable")
    groups = [order[pid[order] == q] for q in np.unique(pid)][:questions]

    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(device).eval()
    load_s = time.time() - t
    L1, D = model.config.num_hidden_layers + 1, model.config.hidden_size
    ck = Clock(sync=torch.cuda.synchronize)
    torch.cuda.reset_peak_memory_stats()
    n_answers = n_tokens = 0
    diffs = {"lotus": 0.0, "act_vit": 0.0}          # max |GPU - CPU| over the answers checked
    for idx in groups:
        batch = [input_ids[i] for i in idx]
        pls = [int(prompt_lens[i]) for i in idx]
        Lmax = max(len(b) for b in batch)
        pad = model.config.eos_token_id or 0
        if isinstance(pad, (list, tuple)):
            pad = pad[0]
        ids = torch.full((len(batch), Lmax), int(pad), dtype=torch.long)
        att = torch.zeros((len(batch), Lmax), dtype=torch.long)
        for j, b in enumerate(batch):
            tt = b if torch.is_tensor(b) else torch.as_tensor(b)
            ids[j, :len(tt)] = tt
            att[j, :len(tt)] = 1
        with ck("forward_shared"), torch.no_grad():
            out = model(ids.to(device), attention_mask=att.to(device), output_hidden_states=True)
        for j in range(len(batch)):
            s, e = pls[j], len(batch[j])
            if e - s == 0:
                continue
            n_answers += 1
            n_tokens += e - s
            with ck("lotus_pooling"):
                h = np.stack([out.hidden_states[i][j, s:e].float().cpu().numpy() for i in WINDOW])
                p_cpu = s57.pool_answer(h, ("core", "static", "velocity"))
            with ck("act_vit_pooling"):
                layers = [out.hidden_states[li][j, s:e].float().cpu().numpy() for li in range(L1)]
                a_cpu = s59.act_pool(np.stack(layers, axis=0), 8, 20)
            if gpu_pool:
                # The same two poolings on the GPU, where the hidden states already are. Only the
                # pooled result crosses to the host, as a production extractor would do it.
                with ck("lotus_pooling_gpu"):
                    hg = torch.stack([out.hidden_states[i][j, s:e] for i in WINDOW]).float()
                    p_gpu = {k: v.cpu().numpy() for k, v in lotus_pool_torch(hg).items()}
                with ck("act_vit_pooling_gpu"):
                    ag = torch.stack([out.hidden_states[li][j, s:e] for li in range(L1)]).float()
                    a_gpu = act_pool_torch(ag, 8, 20).cpu().numpy()
                if n_answers <= 50:
                    diffs["lotus"] = max(diffs["lotus"], max(float(np.abs(p_gpu[k] - p_cpu[k]).max())
                                                             for k in p_gpu))
                    diffs["act_vit"] = max(diffs["act_vit"], float(np.abs(a_gpu - a_cpu).max()))
        del out
    per = {k: round(1000.0 * v / max(1, n_answers), 3) for k, v in ck.t.items()}
    floats = stored_floats_per_answer(D)
    extra = {}
    if gpu_pool:
        extra["max_abs_diff_gpu_vs_cpu_first_50"] = {k: round(v, 6) for k, v in diffs.items()}
    return {**extra, "model_id": model_id, "questions": len(groups), "answers": n_answers,
            "mean_answer_tokens": round(n_tokens / max(1, n_answers), 1),
            "model_load_seconds": round(load_s, 1), "per_answer_ms": per,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
            "hidden_size": int(D), "layers_plus_embedding": int(L1),
            "stored_kb_per_answer_float16": {k: round(2 * v / 1024.0, 1) for k, v in floats.items()}}


def stage_act_vit(dataset, model_folder, data_dir, device, n_eff=20):
    import torch
    s56 = _load("s56", "56_run_method.py")
    import methods as M
    from methods.act_vit import DEFAULT_AT_DIR, EPOCHS, BATCH, LLM_INDEX, build_model, pad_features
    c = M.canonical()
    method = M.get("act_vit")
    method.configure(argparse.Namespace(m_at_dir=DEFAULT_AT_DIR, m_n_eff=n_eff, m_l_eff=8,
                                        m_epochs=EPOCHS))
    t = time.time()
    data = s56.load_bundle(dataset, model_folder, data_dir, device, "bfloat16")
    pre = method.precompute(data)
    load_s = time.time() - t
    rss_after_load = peak_rss_gb()
    n = len(data.labels)
    tr, te = (np.asarray(v, dtype=int) for v in c["question_split"](data.is_known, data.prompt_id, n, SEED))
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    s = method.score(data, pre, tr, te)
    torch.cuda.synchronize()
    train_s = time.time() - t
    peak_train = torch.cuda.max_memory_allocated() / 1024 ** 3

    # Per-answer inference of the same architecture. Timing does not depend on the weights, so a
    # freshly built model gives the inference cost without re-running training.
    at, order = pre["at"], pre["order"]
    model = build_model(at.shape[1], at.shape[2], device).eval()
    idx_t = torch.full((BATCH,), int(LLM_INDEX.get(model_folder, 0)), dtype=torch.long, device=device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        for a in range(0, len(te), BATCH):
            rows = te[a:a + BATCH]
            x = torch.from_numpy(np.asarray(at[order[rows]], dtype=np.float32)).to(device)
            model(pad_features(x), idx_t[:len(rows)])
    torch.cuda.synchronize()
    infer_s = time.time() - t
    return {"tensor_load_seconds": round(load_s, 1), "stored_gb_loaded": round(at.nbytes / 1024 ** 3, 3),
            "peak_rss_gb_after_load": rss_after_load,
            "offline_training_seconds": round(train_s, 1), "epochs": EPOCHS,
            "peak_gpu_gb_training": round(peak_train, 3),
            "per_answer_ms": round(1000.0 * infer_s / len(te), 4),
            "peak_gpu_gb_inference": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_pooled_auroc": float(c["pooled_auroc"](np.asarray(s, dtype=float), data.labels[te])),
            "n_eff": int(at.shape[2]), "l_eff": int(at.shape[1])}


# ---------------------------------------------------------------------------------------------
# CPU: the cross-model mapping
# ---------------------------------------------------------------------------------------------

def time_mapping(Zt, Zx, groups, tr, te, seed):
    s66 = _load("s66", "66_cross_model.py")
    ck = Clock()
    with ck("offline_ridge_fit_with_cv"):
        align = s66.fit_alignment(Zt[tr], Zx[tr], groups, seed)
    with ck("per_answer_apply"):
        s66.ridge_predict(align, Zt[te])
    return {"offline_ridge_fit_with_cv_seconds": round(ck.t["offline_ridge_fit_with_cv"], 3),
            "per_answer_ms": round(1000.0 * ck.t["per_answer_apply"] / len(te), 5),
            "map_parameters": int(align["W"].size + align["my"].size + 2 * align["mx"].size),
            "cv_r2": float(align["cv_r2"]), "n_pairs": int(len(tr))}


def stage_mapping(dataset, source, target, data_dir, cross_dir):
    m = mods()
    s66 = _load("s66", "66_cross_model.py")
    import methods.base as B
    c = B.canonical()
    spec = m["s44"].CONDITION_SPECS[CONDITION]

    def load(folder):
        f, y, pid, known = m["s44"].load_new_dataset(dataset, data_dir, folder, condition=CONDITION)
        return {"feats": f, "y": np.asarray(y, dtype=int), "pid": np.asarray(pid), "known": known}

    S, T = load(source), load(target)
    X = s66.load_cross(cross_dir, source, target, dataset, T["y"], T["pid"])
    tr_t, te_t = (np.asarray(v, dtype=int) for v in c["question_split"](T["known"], T["pid"], len(T["y"]), SEED))
    tr_s, _ = (np.asarray(v, dtype=int) for v in c["question_split"](S["known"], S["pid"], len(S["y"]), SEED))
    t = time.time()
    Zt = m["s61"].build("hosvd", T["feats"], spec, tr_t, SEED)
    both = {k: np.concatenate([S["feats"][k], X[k]], axis=0) for k in S["feats"]}
    Zx = m["s61"].build("hosvd", both, spec, tr_s, SEED)[len(S["y"]):]
    proj_s = time.time() - t
    out = time_mapping(Zt, Zx, T["pid"][tr_t], tr_t, te_t, SEED)
    out["projections_seconds_both_models"] = round(proj_s, 1)
    meta_path = os.path.join(cross_dir, "%s_reads_%s" % (source, target), "%s_window.json" % dataset)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("elapsed_seconds") and meta.get("n_answers"):
            out["source_reading_target_answers_gpu_seconds"] = meta["elapsed_seconds"]
            out["source_reading_ms_per_answer"] = round(
                1000.0 * meta["elapsed_seconds"] / meta["n_answers"], 2)
            out["source_reading_answers"] = meta["n_answers"]
    out.update({"source": source, "target": target})
    return out


# ---------------------------------------------------------------------------------------------

def run_stage(stage, a, data_dir):
    if stage == "extract":
        return stage_extract(a.dataset, a.model_folder, data_dir, a.device, a.questions)
    if stage == "extract_gpu":
        return stage_extract(a.dataset, a.model_folder, data_dir, a.device, a.questions, gpu_pool=True)
    if stage == "fit_lotus":
        return stage_fit("lotus", a.dataset, a.model_folder, data_dir)
    if stage == "fit_flatpca":
        return stage_fit("flat_pca", a.dataset, a.model_folder, data_dir)
    if stage == "act_vit":
        return stage_act_vit(a.dataset, a.model_folder, data_dir, a.device, a.n_eff)
    if stage == "mapping":
        if not a.source:
            raise SystemExit("--stages mapping needs --source (the detector's model); "
                             "--model_folder is the target")
        return stage_mapping(a.dataset, a.source, a.model_folder, data_dir, a.cross_dir)
    raise SystemExit("unknown stage %r (choose from %s)" % (stage, ", ".join(STAGES)))


def self_test():
    print("=" * 78)
    print("  SELF-TEST: 89_cost_profile")
    print("=" * 78)
    m = mods()
    rng = np.random.default_rng(0)
    n, spec = 300, [("core", 2, 4), ("static", 3, 5)]
    feats = {"core": rng.standard_normal((n, 5, 30)).astype(np.float32),
             "static": rng.standard_normal((n, 5, 60)).astype(np.float32)}
    y = (np.arange(n) % 2).astype(int)
    feats["core"][:, 1, :] += y[:, None] * 0.8
    tr, te = np.arange(0, 220), np.arange(220, n)

    # The timed pipelines must produce the reported features, not something that merely resembles them.
    r1, Z1 = time_reduction("lotus", feats, spec, tr, te, y, 3)
    assert np.array_equal(Z1, m["s61"].build("hosvd", feats, spec, tr, 3)), "LOTUS features differ from 61"
    r2, Z2 = time_reduction("flat_pca", feats, spec, tr, te, y, 3)
    assert np.allclose(Z2, m["s61"].build("flat_pca", feats, spec, tr, 3), atol=1e-4), \
        "flat PCA features differ from 61"
    print("  [PASS] timed LOTUS and flat-PCA features equal 61's hosvd and flat_pca arms")

    # Parameter and multiply-add counts follow the closed forms the paper reports.
    assert r1["fitted_basis_parameters"] == (5 * 2 + 30 * 4) + (5 * 3 + 60 * 5)
    assert r2["fitted_basis_parameters"] == (150 * 8 + 150) + (300 * 15 + 300)
    assert r1["projection_macs_per_answer"] < r2["projection_macs_per_answer"]
    for r in (r1, r2):
        for k in ("offline_basis_fit", "offline_classifier_fit", "per_answer_total", "per_answer_ms"):
            assert k in r and r[k] >= 0, k
        assert r["width"] == 8 + 15 and 0.5 < r["test_pooled_auroc"] <= 1.0
    print("  [PASS] parameter and multiply-add counts match their closed forms; every timing is present")

    # Mapping: the ridge fit is 66's, timed on synthetic cores with a known linear relation.
    Zt = rng.standard_normal((n, 12))
    Zx = Zt @ rng.standard_normal((12, 12)) + 0.01 * rng.standard_normal((n, 12))
    r3 = time_mapping(Zt, Zx, (np.arange(n) // 10)[tr], tr, te, 0)
    assert r3["cv_r2"] > 0.99 and r3["map_parameters"] == 12 * 12 + 12 + 24
    print("  [PASS] mapping timing runs 66.fit_alignment (CV R^2 %.4f on a linear relation)" % r3["cv_r2"])

    f = stored_floats_per_answer(3584)
    assert f["lotus"] == 43 * 3584 and abs(f["act_vit_L8_N5"] / f["lotus"] - 40 / 43) < 1e-12
    assert peak_rss_gb() is None or peak_rss_gb() > 0
    print("  [PASS] stored-size arithmetic (ACT-ViT L8 N5 = 0.93x LOTUS) and the peak-memory probe")

    # The GPU poolings timed by extract_gpu must compute what the CPU poolings compute (run here on
    # CPU tensors; the operations are the same on the GPU).
    import torch
    s57 = _load("s57", "57_extract_all_layers.py")
    s59 = _load("s59", "59_extract_act_tensors.py")
    worst_l = worst_a = 0.0
    for L1 in (29, 33, 45):
        for T in (1, 2, 7, 17, 64):
            a = rng.standard_normal((L1, T, 24)).astype(np.float32)
            ref_l = s57.pool_answer(a[WINDOW], ("core", "static", "velocity"))
            got_l = {k: v.numpy() for k, v in lotus_pool_torch(torch.from_numpy(a[WINDOW])).items()}
            worst_l = max(worst_l, max(float(np.abs(got_l[k] - ref_l[k]).max()) for k in ref_l))
            got_a = act_pool_torch(torch.from_numpy(a), 8, 20).numpy()
            worst_a = max(worst_a, float(np.abs(got_a - s59.act_pool(a, 8, 20)).max()))
    assert worst_l < 1e-5, worst_l
    assert worst_a == 0.0, worst_a
    print("  [PASS] torch poolings equal 57.pool_answer (max diff %.1e) and 59.act_pool (exact) for 29/33/45 "
          "layers and 1-64 tokens" % worst_l)

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset", default="truthfulqa")
    p.add_argument("--model_folder")
    p.add_argument("--source", default=None, help="mapping stage: the detector's model")
    p.add_argument("--stages", nargs="+", default=None, choices=STAGES)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--cross-dir", default=os.path.abspath(os.path.join(HERE, "..", "data-crossread")))
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--questions", type=int, default=100, help="extract stage: questions to time")
    p.add_argument("--n-eff", type=int, default=20, help="act_vit stage: ACT-ViT token slots")
    a = p.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.model_folder or not a.stages:
        raise SystemExit("--model_folder and --stages are required (or use --self-test)")
    data_dir = a.data_dir
    if not data_dir:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]

    if len(a.stages) > 1:
        # One process per stage, so each stage's peak memory is its own.
        failed = []
        for st in a.stages:
            cmd = [sys.executable, os.path.abspath(__file__), "--stages", st] + [
                x for x in sys.argv[1:] if x not in a.stages and x != "--stages"]
            print("\n==> %s" % " ".join(cmd), flush=True)
            if subprocess.call(cmd) != 0:
                failed.append(st)
        if failed:
            raise SystemExit("stages failed: %s" % ", ".join(failed))
        return

    stage = a.stages[0]
    t0 = time.time()
    res = run_stage(stage, a, data_dir)
    res.update({"stage": stage, "dataset": a.dataset, "model_folder": a.model_folder,
                "peak_rss_gb_process": peak_rss_gb(), "wall_seconds": round(time.time() - t0, 1),
                "complete": True})
    os.makedirs(a.out_dir, exist_ok=True)
    tag = ("_from_%s" % a.source) if stage == "mapping" else ""
    dst = os.path.join(a.out_dir, "cost_%s_%s_%s%s.json" % (stage, a.model_folder, a.dataset, tag))
    with open(dst, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    print("wrote %s" % dst)


if __name__ == "__main__":
    main()
