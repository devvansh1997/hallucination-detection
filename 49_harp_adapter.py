"""
49_harp_adapter.py -- convert our pinned generations into HARP's input format
=====================================================================================================
Purpose: RERUN HARP's released code on our data, not reimplement their method. We author only this
adapter; every scoring decision -- hidden-state extraction, the lm_head SVD, the subspace
projection, the MLP, the AUROC -- stays in their code, untouched.

HOW THE HANDOFF WORKS
    HARP's main.py caches its generation+judging step:

        if os.path.exists(known_save_path) and os.path.exists(unknown_save_path):
            print("known and unknown file exists, skip...")          # main.py:120-123

    so writing ./dataset/[known]{ds}.jsonl and ./dataset/[unknown]{ds}.jsonl makes their pipeline
    skip generation entirely and run their method on whatever we put there. That is the whole
    integration surface: two files.

THEIR SCHEMA (get_known.py: DatasetJudgeResult / BeamSearchResult), one JSON object per line:
    ind                : int    -- dataset index
    prompt             : str    -- FULL prompt text, including the passage for tydiqa
    correct_answers    : [str]
    incorrect_answers  : [str]
    result             : [ {beam_id: int, gen_text: str, score: {is_correct: bool, ...}} ]
  A prompt is "known" if ANY beam has score.is_correct, else "unknown". Their main.py reads only
  prompt / result / gen_text / score.is_correct downstream; the four similarity floats inside
  score are diagnostics from their judge and are never re-read, so we emit them as null rather
  than inventing values (our Phase 1 persisted the boolean only).

gen_text IS DECODED FRESH, NOT TAKEN FROM decoded_text. Our Phase 1 saves decoded_text .strip()'d
for readability, but HARP re-tokenizes prompt+gen_text, and Llama/Qwen BPE bakes the leading space
into word-initial tokens -- so a stripped string retokenizes differently. We decode the pinned
completion ids with skip_special_tokens=True and no strip, which is exactly what their
get_known.py does. (Same class of bug we hit in 40_validate_dataset.py's round-trip check.)

WRITES NOTHING INTO HARP-Code/ BY DEFAULT. Output goes to a per-model staging directory; the copy
into their ./dataset/ is a separate explicit step, because their cache path is NOT model-specific
(main.py:80-81 keys only on ds_name) and staging both models in one place would let a Qwen run
silently consume LLaMA's generations.

Usage:
  python 49_harp_adapter.py --self-test
  python 49_harp_adapter.py --dataset triviaqa --model_folder qwen-2.5-7b-instruct
  python 49_harp_adapter.py --dataset all --model_folder qwen-2.5-7b-instruct
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# our dataset name -> the name HARP's main.py expects (its DATASET_PATH / PROMPT_TEMPLATE keys)
DS_OURS_TO_HARP = {"truthfulqa": "truthful_qa", "triviaqa": "trivia_qa",
                   "nq_open": "nq_open", "tydiqa_gp": "tydiqa"}
# our model folder -> their MODEL_PATH key
MODEL_OURS_TO_HARP = {"qwen-2.5-7b-instruct": "Qwen2.5-7B-Instruct",
                      "llama-3.1-8b-instruct": "Llama-3.1-8B"}
BEAMS_PER_PROMPT = 10


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def group_beams_by_prompt(prompt_ids):
    """Row indices of the sequences file, grouped by prompt, order preserved. Phase 1 writes beams
    contiguously per prompt, but we group explicitly rather than assuming stride-10 blocks."""
    order, groups = [], {}
    for row, p in enumerate(prompt_ids):
        p = int(p)
        if p not in groups:
            groups[p] = []
            order.append(p)
        groups[p].append(row)
    return order, groups


def build_record(ind, prompt_text, correct_answers, incorrect_answers, gen_texts, is_correct_flags):
    """One HARP DatasetJudgeResult as a plain dict. score's similarity fields are null: their judge
    produced them, ours did not persist them, and their main.py never reads them back."""
    return {
        "ind": int(ind),
        "prompt": prompt_text,
        "correct_answers": list(correct_answers),
        "incorrect_answers": list(incorrect_answers),
        "result": [
            {"beam_id": int(b), "gen_text": t,
             "score": {"is_correct": bool(c), "max_correct_sen_sim": None,
                       "max_correct_rouge_sim": None, "max_incorrect_sen_sim": None,
                       "max_incorrect_rouge_sim": None}}
            for b, (t, c) in enumerate(zip(gen_texts, is_correct_flags))
        ],
    }


def is_known(record):
    """HARP's DatasetJudgeResult.is_known(): any beam judged correct."""
    return any(b["score"]["is_correct"] for b in record["result"])


def convert(dataset, model_folder, data_dir, out_root, verbose=True):
    import torch
    from transformers import AutoTokenizer
    import yaml

    seq_path = os.path.join(data_dir, model_folder, f"{dataset}_sequences_v1.pt")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(
            f"{seq_path} not found. Only datasets generated through 39_generate_dataset.py have a "
            f"pinned sequences file. LLaMA/TruthfulQA came from the older Route-N path and stores "
            f"pooled features only -- no decoded text -- so it cannot be adapted without first "
            f"regenerating it through 39 (~30 min GPU for 817 prompts).")

    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ds_cfg = next(d for d in cfg["datasets"] if d["name"] == dataset)
    model_id = next(m["id"] for m in cfg["models"] if m["folder"] == model_folder)

    seq = torch.load(seq_path, weights_only=False)
    input_ids, prompt_lens = seq["input_ids"], seq["prompt_len"]
    prompt_ids, halluc_flags = seq["prompt_id"], seq["all_hallucination_flag"]

    # answers come from the SOURCE dataset, via the same loader Phase 1 used, so prompt_id indexes
    # line up by construction rather than by assumption
    gen_mod = _load("s06_gen", "39_generate_dataset.py")
    samples, _ = gen_mod.load_dataset_samples(ds_cfg)
    by_pid = {s["prompt_id"]: s for s in samples}

    tok = AutoTokenizer.from_pretrained(model_id)
    order, groups = group_beams_by_prompt(prompt_ids)

    known, unknown = [], []
    for pid in order:
        rows = groups[pid]
        sample = by_pid[pid]
        gen_texts, flags = [], []
        for r in rows:
            comp_ids = input_ids[r][prompt_lens[r]:].tolist()
            # no .strip() -- see module docstring
            gen_texts.append(tok.decode(comp_ids, skip_special_tokens=True))
            flags.append(not bool(halluc_flags[r]))   # ours stores hallucinated; theirs is_correct
        rec = build_record(pid, sample["prompt_text"], sample["correct_answers"],
                           sample["incorrect_answers"], gen_texts, flags)
        (known if is_known(rec) else unknown).append(rec)

    # -- validate against what Phase 1 recorded, so a silent mis-grouping cannot slip through --
    n_beams_out = sum(len(r["result"]) for r in known + unknown)
    assert n_beams_out == len(input_ids), \
        f"{dataset}: emitted {n_beams_out} beams but the sequences file has {len(input_ids)}"
    bad = [r["ind"] for r in known + unknown if len(r["result"]) != BEAMS_PER_PROMPT]
    assert not bad, f"{dataset}: {len(bad)} prompts do not have {BEAMS_PER_PROMPT} beams: {bad[:5]}"
    if "all_is_known" in seq:
        ours_known = int(np.sum(np.asarray(seq["all_is_known"], dtype=bool)))
        assert len(known) == ours_known, (
            f"{dataset}: our known count is {ours_known} but the adapter produced {len(known)}. "
            f"The known/unknown split must be reproduced exactly or the train/test partition "
            f"differs from ours.")
    rate = float(np.asarray(halluc_flags, dtype=bool).mean() * 100)

    out_dir = os.path.join(out_root, model_folder)
    harp_ds = DS_OURS_TO_HARP[dataset]
    paths = {"known": os.path.join(out_dir, f"[known]{harp_ds}.jsonl"),
             "unknown": os.path.join(out_dir, f"[unknown]{harp_ds}.jsonl")}
    os.makedirs(out_dir, exist_ok=True)
    for kind, recs in (("known", known), ("unknown", unknown)):
        with open(paths[kind], "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    info = {"dataset": dataset, "harp_dataset": harp_ds, "model_folder": model_folder,
            "harp_model_key": MODEL_OURS_TO_HARP.get(model_folder, "UNMAPPED"),
            "n_prompts": len(order), "n_beams": n_beams_out, "n_known": len(known),
            "n_unknown": len(unknown), "hallucination_rate_pct": round(rate, 2),
            "source_sequences": os.path.abspath(seq_path), "written": paths}
    if verbose:
        print(f"  [{dataset}] {len(order)} prompts / {n_beams_out} beams  "
              f"known={len(known)} unknown={len(unknown)}  halluc={rate:.1f}%", flush=True)
        for p in paths.values():
            print(f"      wrote {p}", flush=True)
    return info


def self_test():
    print("=" * 70)
    print("  SELF-TEST: 49_harp_adapter (synthetic, no model, no cluster files)")
    print("=" * 70)

    pids = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    order, groups = group_beams_by_prompt(pids)
    assert order == [0, 1, 2] and groups == {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
    inter = [0, 1, 0, 1]
    o2, g2 = group_beams_by_prompt(inter)
    assert o2 == [0, 1] and g2 == {0: [0, 2], 1: [1, 3]}, \
        "grouping must not assume beams are contiguous per prompt"
    print("  [PASS] group_beams_by_prompt: contiguous and interleaved layouts both handled")

    rec = build_record(7, "Q: x\nA:", ["a", "b"], ["c"], ["ans0", "ans1"], [False, True])
    assert rec["ind"] == 7 and len(rec["result"]) == 2
    assert [b["beam_id"] for b in rec["result"]] == [0, 1]
    assert rec["result"][1]["score"]["is_correct"] is True
    assert rec["result"][0]["score"]["max_correct_sen_sim"] is None
    assert is_known(rec) is True
    print("  [PASS] build_record: HARP schema, beam ids sequential, null similarity fields")

    all_wrong = build_record(0, "p", ["a"], [], ["x", "y"], [False, False])
    assert is_known(all_wrong) is False
    all_right = build_record(0, "p", ["a"], [], ["x", "y"], [True, True])
    assert is_known(all_right) is True
    one_right = build_record(0, "p", ["a"], [], ["x", "y"], [False, True])
    assert is_known(one_right) is True, "a single correct beam makes a prompt known"
    print("  [PASS] is_known: matches HARP's any-correct rule on all-wrong / all-right / mixed")

    # the record must survive a JSON round-trip exactly as their jsonlload will read it
    line = json.dumps(rec, ensure_ascii=False)
    back = json.loads(line)
    assert back["prompt"] == rec["prompt"]
    assert back["result"][1]["gen_text"] == "ans1"
    assert back["result"][1]["score"]["is_correct"] is True
    assert "\n" not in line, "each record must occupy exactly one jsonl line"
    print("  [PASS] json round-trip: single line, fields readable exactly as main.py reads them")

    # non-ASCII content must survive (ensure_ascii=False, matching their writer)
    uni = build_record(1, "Q: Wer ist Gauss?\nA:", ["Carl Friedrich Gauss"], [],
                       ["Carl Friedrich Gauss war ein Mathematiker."], [True])
    assert json.loads(json.dumps(uni, ensure_ascii=False))["result"][0]["gen_text"].endswith("."), \
        "unicode content must round-trip"
    print("  [PASS] unicode content round-trips under ensure_ascii=False")

    assert set(DS_OURS_TO_HARP) == {"truthfulqa", "triviaqa", "nq_open", "tydiqa_gp"}
    assert DS_OURS_TO_HARP["tydiqa_gp"] == "tydiqa" and DS_OURS_TO_HARP["triviaqa"] == "trivia_qa"
    print("  [PASS] dataset name mapping covers all four and matches HARP's keys")

    print("\n[PASS] All self-test assertions passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=list(DS_OURS_TO_HARP) + ["all"])
    ap.add_argument("--model_folder", default="qwen-2.5-7b-instruct")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-root", default=os.path.join(HERE, "harp_input"),
                    help="Staging dir. Deliberately NOT inside HARP-Code/ -- their cache path is "
                         "not model-specific, so both models' files would collide there.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return

    if a.model_folder not in MODEL_OURS_TO_HARP:
        print(f"WARNING: {a.model_folder} has no HARP MODEL_PATH mapping; files will still be "
              f"written but you must pick --model_name yourself when invoking their main.py.")

    if a.data_dir:
        data_dir = a.data_dir
    else:
        import yaml
        with open(os.path.join(HERE, "config.yaml")) as f:
            data_dir = yaml.safe_load(f)["output"]["data_dir"]

    datasets = list(DS_OURS_TO_HARP) if a.dataset == "all" else [a.dataset]
    out_dir = os.path.join(a.out_root, a.model_folder)
    print("=" * 78)
    print("  This run will WRITE (and nothing else):")
    for ds in datasets:
        h = DS_OURS_TO_HARP[ds]
        print(f"    {os.path.join(out_dir, f'[known]{h}.jsonl')}")
        print(f"    {os.path.join(out_dir, f'[unknown]{h}.jsonl')}")
    print("  Read-only: ../data/** and HARP-Code/** are never modified.")
    print("=" * 78, flush=True)

    infos, failed = [], []
    for ds in datasets:
        try:
            infos.append(convert(ds, a.model_folder, data_dir, a.out_root))
        except FileNotFoundError as e:
            print(f"  [{ds}] SKIPPED -- {e}", flush=True)
            failed.append(ds)

    summary_path = os.path.join(out_dir, "adapter_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"converted": infos, "skipped": failed,
                   "harp_model_key": MODEL_OURS_TO_HARP.get(a.model_folder)}, f, indent=2)
    print(f"\nWrote: {summary_path}")

    if infos:
        key = MODEL_OURS_TO_HARP.get(a.model_folder, "<MODEL_NAME>")
        print("\nNext, from inside HARP-Code/ -- copy ONE model's files in at a time, because")
        print("their ./dataset/ cache path is keyed on dataset only, not on model:")
        print(f"    cp {out_dir}/*.jsonl ./dataset/")
        for i in infos:
            print(f"    python ./main.py --train_ratio=0.75 --train_batch_size=128 "
                  f"--train_epochs=50 --model_name=\"{key}\" --dataset_name=\"{i['harp_dataset']}\"")


if __name__ == "__main__":
    main()
