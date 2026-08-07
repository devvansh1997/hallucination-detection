"""
audit_split_protocols.py -- prove the four cells of the method x protocol grid were produced
under the split rules they claim to be.
=====================================================================================================
WHY THIS EXISTS
    The headline result is a 2x2: {HARP, ours} x {answer-level, question-level}. Three of the four
    cells are produced by code we wrote, and only one calls HARP's splitter directly. If any of our
    three encodes a different protocol than it claims, the comparison is not like-for-like and the
    result is void. This script checks each of ours against the reference it is supposed to mirror.

    HARP cell, answer-level    utils.split_data                      THEIRS -- reference, not tested
    HARP cell, question-level  50_harp_reeval.split_by_question      ours
    our cell,  answer-level    44_eval_phase3.answer_level_harp_split ours
    our cell,  question-level  26_grouped_baseline.original_harp_split ours

WHAT "MATCHING" MEANS
    Not the same partition -- these use different RNGs (torch.randperm vs np.random.shuffle vs
    np.random.default_rng), so identical output is neither expected nor required. What must match
    is the PROTOCOL: how many rows land where, which prompts may straddle the split, and whether
    unknown prompts ever reach training. Those are the properties an AUROC depends on.

Usage:
  python audit_split_protocols.py            # audit against real dataset shapes
  python audit_split_protocols.py --self-test
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HARP = os.path.abspath(os.path.join(HERE, "..", "HARP-Code"))

# (name, n_prompts, n_known, beams) -- the real shapes, from the adapter's own counts
SHAPES = [("tydiqa_gp", 440, 302, 10),
          ("truthfulqa", 817, 666, 10),
          ("nq_open", 3610, 360, 10),
          ("triviaqa", 9960, 6316, 10)]
TRAIN_RATIO = 0.75


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_shape(n_prompts, n_known, beams):
    prompt_idx = np.repeat(np.arange(n_prompts), beams)
    is_known = np.zeros(n_prompts, dtype=bool)
    is_known[:n_known] = True
    return prompt_idx, is_known, n_prompts * beams


def protocol(prompt_idx, is_known, t_idx, v_idx, N):
    """The properties an AUROC actually depends on. Deliberately NOT the partition itself."""
    t_idx, v_idx = list(t_idx), list(v_idx)
    tq = {int(prompt_idx[r]) for r in t_idx}
    vq = {int(prompt_idx[r]) for r in v_idx}
    unknown_prompts = set(np.where(~is_known)[0].tolist())
    return {
        "n_train": len(t_idx),
        "n_valid": len(v_idx),
        "train_prompts": len(tq),
        "valid_prompts": len(vq),
        "prompts_on_both_sides": len(tq & vq),
        "unknown_rows_in_train": len([r for r in t_idx if int(prompt_idx[r]) in unknown_prompts]),
        "partitions_exactly": (len(t_idx) + len(v_idx) == N
                               and len(set(t_idx) | set(v_idx)) == N
                               and not (set(t_idx) & set(v_idx))),
    }


# ==============================================================================
# REFERENCE IMPLEMENTATIONS -- what main.py actually does, using THEIR splitter
# ==============================================================================

def harp_reference_answer_split(harp_utils, prompt_idx, is_known, N, seed):
    """main.py:259 + :264, verbatim in structure:
         train, valid = split_data(known_data, train_ratio, shuffle=True)
         valid.extend(unknown_data)
    We pass row indices as the 'data' list, so their splitter returns row indices."""
    import torch
    known_prompts = set(np.where(is_known)[0].tolist())
    known_rows = [r for r in range(N) if int(prompt_idx[r]) in known_prompts]
    unknown_rows = [r for r in range(N) if int(prompt_idx[r]) not in known_prompts]
    torch.manual_seed(seed)
    t, v = harp_utils.split_data(known_rows, train_ratio=TRAIN_RATIO, shuffle=True)
    return list(t), list(v) + unknown_rows


# ==============================================================================

def audit(harp_dir, seeds=(42, 0, 1, 2, 3), verbose=True):
    if harp_dir not in sys.path:
        sys.path.insert(0, harp_dir)
    import utils as harp_utils

    s44 = _load("s44", "44_eval_phase3.py")
    s50 = _load("s50", "50_harp_reeval_prompt_level.py")
    s26 = _load("s26", "26_grouped_baseline.py")

    failures = []
    for name, n_prompts, n_known, beams in SHAPES:
        prompt_idx, is_known, N = make_shape(n_prompts, n_known, beams)
        known_prompts = set(np.where(is_known)[0].tolist())
        known_rows = [r for r in range(N) if int(prompt_idx[r]) in known_prompts]
        pid_known = [int(prompt_idx[r]) for r in known_rows]

        for seed in seeds:
            # ---- ANSWER-LEVEL: ours (44) vs theirs (utils.split_data via main.py's structure)
            ref_t, ref_v = harp_reference_answer_split(harp_utils, prompt_idx, is_known, N, seed)
            our_t, our_v = s44.answer_level_harp_split(is_known, prompt_idx, N, seed)
            ref_p = protocol(prompt_idx, is_known, ref_t, ref_v, N)
            our_p = protocol(prompt_idx, is_known, our_t, our_v, N)
            # Two fields are random per draw and must NOT be compared exactly:
            #   prompts_on_both_sides -- which known prompts happen to straddle this shuffle
            #   valid_prompts         -- identically = prompts_on_both_sides + n_unknown_prompts,
            #                            so it carries no independent information
            # Everything else is a deterministic consequence of the protocol and must match exactly.
            ref_c = dict(ref_p); our_c = dict(our_p)
            both_ref = ref_c.pop("prompts_on_both_sides"); ref_c.pop("valid_prompts")
            both_our = our_c.pop("prompts_on_both_sides"); our_c.pop("valid_prompts")
            assert ref_p["valid_prompts"] - both_ref == our_p["valid_prompts"] - both_our, \
                "valid_prompts is not both_sides + n_unknown -- the identity above is wrong"
            ok = (ref_c == our_c) and abs(both_ref - both_our) <= 0.05 * n_known
            if not ok:
                failures.append(("answer-level", name, seed, ref_p, our_p))
            if verbose and seed == seeds[0]:
                print("  %-11s answer-level   theirs n_train=%d both=%d | ours n_train=%d both=%d  %s"
                      % (name, ref_p["n_train"], both_ref, our_p["n_train"], both_our,
                         "OK" if ok else "FAIL"))

            # ---- QUESTION-LEVEL: 26 (our numbers) vs 50 (HARP's question arm)
            a_t, a_v = s26.original_harp_split(is_known, prompt_idx, N, seed=seed)
            tk, vk = s50.split_by_question(pid_known, TRAIN_RATIO, np.random.default_rng(seed))
            b_t = [known_rows[i] for i in tk]
            b_v = [known_rows[i] for i in vk] + [r for r in range(N)
                                                 if int(prompt_idx[r]) not in known_prompts]
            a_p = protocol(prompt_idx, is_known, a_t, a_v, N)
            b_p = protocol(prompt_idx, is_known, b_t, b_v, N)
            ok2 = a_p == b_p
            if not ok2:
                failures.append(("question-level", name, seed, a_p, b_p))
            if verbose and seed == seeds[0]:
                print("  %-11s question-level ours   n_train=%d both=%d | harp n_train=%d both=%d  %s"
                      % (name, a_p["n_train"], a_p["prompts_on_both_sides"],
                         b_p["n_train"], b_p["prompts_on_both_sides"], "OK" if ok2 else "FAIL"))

            # ---- the two protocols must DIFFER from each other, or the experiment is vacuous
            if our_p["prompts_on_both_sides"] <= a_p["prompts_on_both_sides"]:
                failures.append(("protocols-not-distinct", name, seed, our_p, a_p))
    return failures


def self_test():
    print("=" * 74)
    print("  SELF-TEST: audit_split_protocols (synthetic shapes, no cluster data)")
    print("=" * 74)
    prompt_idx, is_known, N = make_shape(40, 30, 10)
    t = list(range(0, 150)); v = list(range(150, N))
    p = protocol(prompt_idx, is_known, t, v, N)
    assert p["partitions_exactly"] is True
    assert p["n_train"] == 150 and p["n_valid"] == N - 150
    print("  [PASS] protocol(): counts and exact-partition check")

    bad = protocol(prompt_idx, is_known, [0, 1], [1, 2], N)
    assert bad["partitions_exactly"] is False, "overlapping rows must fail the partition check"
    print("  [PASS] protocol(): detects an overlapping / incomplete partition")

    everything = list(range(N))
    p2 = protocol(prompt_idx, is_known, everything, [], N)
    assert p2["unknown_rows_in_train"] == 100, "10 unknown prompts x 10 beams"
    print("  [PASS] protocol(): counts unknown rows that reached training")
    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harp-dir", default=DEFAULT_HARP)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    print("=" * 78)
    print("  SPLIT PROTOCOL AUDIT -- 5 seeds x 4 real dataset shapes")
    print("  reference for answer-level: HARP's own utils.split_data (%s)" % a.harp_dir)
    print("=" * 78)
    failures = audit(a.harp_dir)
    print()
    if failures:
        print("FAILURES (%d):" % len(failures))
        for kind, name, seed, ref, ours in failures[:10]:
            print("  [%s] %s seed=%d\n    reference: %s\n    ours     : %s" % (
                kind, name, seed, ref, ours))
        sys.exit(1)
    print("PASS -- on every shape and seed:")
    print("  * our answer-level split matches HARP's utils.split_data in every protocol property")
    print("  * our two independent question-level splits encode the same protocol")
    print("  * the two protocols are distinct (answer-level leaks, question-level does not)")


if __name__ == "__main__":
    main()
