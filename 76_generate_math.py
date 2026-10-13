"""
76_generate_math.py -- sampled chain-of-thought answers to math questions, graded by their final number (T-025).
==================================================================================================

WHY. The token axis can only carry a temporal signal in answers that go wrong partway through (T-023, T-024).
RAGTruth showed such a signal but with a different prompt behind every hallucinated and every correct answer,
and with locations that only human annotators could supply. Math reasoning removes both problems: K sampled
answers to ONE question can be compared with each other, and each is right or wrong by its final number.

WHAT IT DOES. For each question: K sampled answers (plain temperature sampling, no beams), each cut at the
model's own end-of-answer token, graded against the reference number.

    correct      the final number, taken from the last '#### <number>' line, equals the reference
    wrong        it does not, or no number could be read (these are also counted separately)
    truncated    the answer reached max_new_tokens without ending; flagged, never silently graded

A question is MIXED when its K answers include both outcomes. Only mixed questions support the
within-question comparison, so their count is the number this script exists to report first.

WHAT IS WRITTEN (config.yaml math.out_dir / <model folder> /):
    <dataset>[_pilotN]_sequences_v1.pt   same keys the QA pipeline reads -- input_ids, prompt_len, prompt_id,
                                         decoded_text, all_hallucination_flag (1 = wrong), all_is_known,
                                         gen_seeds, decoding_config, git_commit -- plus pred, gold,
                                         extract_method, truncated, question_index
    <dataset>[_pilotN]_summary.json      accuracy, mixed questions, answer lengths, how answers were read

DIFFERENCES FROM 39, on purpose. 39 stops an answer at its first sentence end and decodes with beams, which
is what makes the QA answers one sentence long and near-duplicates of each other. Here the only stops are
end-of-answer tokens and the K answers are independent samples. Seeds are per question (39's scheme), so a
pilot on the first N questions is reproduced by the full run.

Instruction-tuned models only for now: the prompt is the model's chat template around one user turn. A base
model needs a few-shot prompt, which is not built yet; the script refuses rather than improvising one.

  python 76_generate_math.py --self-test
  python 76_generate_math.py --prefetch                                   # dataset into the cache; no GPU
  python 76_generate_math.py --model_folder qwen-2.5-7b-instruct --limit-questions 200
  python 76_generate_math.py --model_folder qwen-2.5-7b-instruct
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>")
NUMBER = r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
MARKED = re.compile(r"####\s*\$?\s*(" + NUMBER + r")")
ANY_NUMBER = re.compile(NUMBER)


# ---------------------------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------------------------

def to_number(s):
    """'1,234.50' -> 1234.5; None if it is not a number."""
    try:
        return float(str(s).replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


def reference_number(answer_field):
    """GSM8K stores the worked solution followed by '#### <number>'."""
    m = MARKED.findall(answer_field)
    return to_number(m[-1]) if m else None


def read_final_number(text):
    """(number, how it was read): the last '#### <number>' if the answer has one ('marker'), else the last
    number anywhere in the text ('last_number'), else (None, 'none')."""
    m = MARKED.findall(text)
    if m:
        return to_number(m[-1]), "marker"
    nums = [n for n in ANY_NUMBER.findall(text) if to_number(n) is not None]
    if nums:
        return to_number(nums[-1]), "last_number"
    return None, "none"


def same_number(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


def cut_at_stop(ids, stop_ids):
    """(tokens before the first stop token, whether a stop token was found)."""
    for i, t in enumerate(ids):
        if int(t) in stop_ids:
            return list(ids[:i]), True
    return list(ids), False


def stop_token_ids(tok):
    """The tokenizer's own end-of-answer token plus any of STOP_TOKENS that it registers as a SPECIAL token.
    The special check matters: a string like '</s>' can also be an ordinary text token in a BPE vocabulary
    (id 128247 in Qwen2.5's), and the pilot of 2026-10-13 carried that id as a stop by mistake."""
    special = set(int(i) for i in tok.all_special_ids) | set(int(i) for i in getattr(tok, "added_tokens_encoder", {}).values())
    ids = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()
    for s in STOP_TOKENS:
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i in special:
            ids.add(i)
    return ids


def question_seed(global_seed, prompt_id):
    """39's scheme: sha256, so a pilot and the full run draw the same answers for the same question."""
    return int(hashlib.sha256(("%s:%s" % (global_seed, prompt_id)).encode()).hexdigest(), 16) % (2 ** 31)


def summarize(prompt_id, wrong, truncated, methods, lengths, k):
    """Accuracy, mixed questions and how the answers were read. wrong: 1 = wrong answer."""
    prompt_id, wrong = np.asarray(prompt_id), np.asarray(wrong).astype(int)
    n_correct = np.array([int((wrong[prompt_id == q] == 0).sum()) for q in np.unique(prompt_id)])
    lengths = np.asarray(lengths)
    return {
        "questions": int(len(n_correct)), "answers": int(len(wrong)), "answers_per_question": int(k),
        "answer_accuracy": float(1.0 - wrong.mean()),
        "questions_all_correct": int((n_correct == k).sum()),
        "questions_all_wrong": int((n_correct == 0).sum()),
        "questions_mixed": int(((n_correct > 0) & (n_correct < k)).sum()),
        "wrong_answers_in_mixed_questions": int(sum(k - c for c in n_correct if 0 < c < k)),
        "correct_per_question_histogram": np.bincount(n_correct, minlength=k + 1).tolist(),
        "truncated_answers": int(np.sum(truncated)),
        "read_by": {m: int(sum(x == m for x in methods)) for m in ("marker", "last_number", "none")},
        "answer_tokens": {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                          "p90": float(np.percentile(lengths, 90)), "max": int(lengths.max())},
    }


# ---------------------------------------------------------------------------------------------

def math_cfg():
    import yaml
    with open(os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg["math"]


def load_questions(ds):
    from datasets import load_dataset
    rows = load_dataset(ds["hf_path"], ds["hf_config"], split=ds["split"])
    out = []
    for i, r in enumerate(rows):
        gold = reference_number(r["answer"])
        if gold is None:
            raise SystemExit("question %d has no reference number" % i)
        out.append({"question_index": i, "question": r["question"].strip(), "gold": gold})
    return out


def git_head():
    try:
        return subprocess.run(["git", "-C", HERE, "log", "--oneline", "-1"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def run(model_folder, dataset, limit, global_seed, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg, mc = math_cfg()
    ds = next((d for d in mc["datasets"] if d["name"] == dataset), None)
    if ds is None:
        raise SystemExit("unknown dataset %r; config.yaml math.datasets has %s" % (dataset, [d["name"] for d in mc["datasets"]]))
    model_id = next((m["id"] for m in cfg["models"] if m["folder"] == model_folder), None)
    if model_id is None:
        raise SystemExit("unknown model folder %r" % model_folder)
    g = mc["generation"]
    k = int(g["num_return_sequences"])
    out_dir = os.path.normpath(os.path.join(HERE, mc["out_dir"], model_folder))
    stem = "%s%s" % (dataset, "_pilot%d" % limit if limit else "")
    seq_path = os.path.join(out_dir, "%s_sequences_v1.pt" % stem)
    if os.path.exists(seq_path):
        raise SystemExit("refusing: %s exists" % seq_path)

    questions = load_questions(ds)
    if limit:
        questions = questions[:limit]
    tok = AutoTokenizer.from_pretrained(model_id)
    if not tok.chat_template:
        raise SystemExit("%s has no chat template: a base model needs a few-shot prompt, which is not built yet" % model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
    model.eval()
    stop_ids = stop_token_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print("  [%s/%s] %d questions x %d answers | stop tokens %s | temperature %s, top_p %s, top_k %s, max_new_tokens %d"
          % (model_folder, stem, len(questions), k, sorted(stop_ids), g["temperature"], g["top_p"], g["top_k"],
             g["max_new_tokens"]), flush=True)

    S = {name: [] for name in ("input_ids", "prompt_len", "prompt_id", "decoded_text", "all_hallucination_flag",
                               "gen_seeds", "pred", "gold", "extract_method", "truncated", "question_index")}
    is_known, lengths = [], []
    t0 = time.time()
    for pid, q in enumerate(questions):
        seed = question_seed(global_seed, pid)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        text = tok.apply_chat_template([{"role": "user", "content": mc["instruction"] + "\n\n" + q["question"]}],
                                       tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        n_prompt = int(inputs.input_ids.shape[1])
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=True, temperature=float(g["temperature"]),
                                 top_p=float(g["top_p"]), top_k=int(g["top_k"]),
                                 repetition_penalty=float(g["repetition_penalty"]), num_return_sequences=k,
                                 max_new_tokens=int(g["max_new_tokens"]), eos_token_id=sorted(stop_ids),
                                 pad_token_id=pad_id)
        any_correct = False
        for b in range(out.shape[0]):
            comp, ended = cut_at_stop(out[b, n_prompt:].tolist(), stop_ids)
            answer = tok.decode(comp, skip_special_tokens=True).strip()
            pred, how = read_final_number(answer)
            correct = same_number(pred, q["gold"])
            any_correct = any_correct or correct
            S["input_ids"].append(torch.tensor(inputs.input_ids[0].tolist() + comp, dtype=torch.long))
            S["prompt_len"].append(n_prompt)
            S["prompt_id"].append(pid)
            S["decoded_text"].append(answer)
            S["all_hallucination_flag"].append(not correct)
            S["gen_seeds"].append(seed)
            S["pred"].append(pred)
            S["gold"].append(q["gold"])
            S["extract_method"].append(how)
            S["truncated"].append(not ended)
            S["question_index"].append(q["question_index"])
            lengths.append(len(comp))
        is_known.append(any_correct)
        if (pid + 1) % 25 == 0 or pid + 1 == len(questions):
            el = time.time() - t0
            wrong = np.asarray(S["all_hallucination_flag"])
            print("    %d/%d questions (%.0fs, ~%.0fs left) | answer accuracy %.3f | mean answer %d tokens"
                  % (pid + 1, len(questions), el, el / (pid + 1) * (len(questions) - pid - 1), 1 - wrong.mean(),
                     int(np.mean(lengths))), flush=True)

    summary = summarize(S["prompt_id"], S["all_hallucination_flag"], S["truncated"], S["extract_method"], lengths, k)
    decoding_config = dict(g, global_seed=global_seed, seed_scheme="sha256(global_seed:prompt_id)",
                           instruction=mc["instruction"], chat_template="tokenizer default, one user turn",
                           stop_token_ids=sorted(stop_ids), dataset=ds, limit_questions=limit or None)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(S, all_is_known=is_known, decoding_config=decoding_config, git_commit=git_head()), seq_path)
    with open(os.path.join(out_dir, "%s_summary.json" % stem), "w") as f:
        json.dump(dict(summary, model_folder=model_folder, dataset=dataset, decoding_config=decoding_config,
                       elapsed_seconds=round(time.time() - t0, 1)), f, indent=1)

    print("\n  SUMMARY %s / %s" % (model_folder, stem))
    print("    answers correct: %.1f%% of %d" % (100 * summary["answer_accuracy"], summary["answers"]))
    print("    questions: %d mixed, %d all correct, %d all wrong (of %d)" % (
        summary["questions_mixed"], summary["questions_all_correct"], summary["questions_all_wrong"],
        summary["questions"]))
    print("    wrong answers inside mixed questions: %d" % summary["wrong_answers_in_mixed_questions"])
    print("    correct answers per question, 0..%d: %s" % (k, summary["correct_per_question_histogram"]))
    a = summary["answer_tokens"]
    print("    answer length in tokens: median %d, mean %d, 90th pct %d, max %d" % (a["median"], a["mean"], a["p90"], a["max"]))
    print("    final number read from: %s | truncated answers: %d" % (summary["read_by"], summary["truncated_answers"]))
    print("  wrote %s" % seq_path)


# ---------------------------------------------------------------------------------------------

def inspect(model_folder, dataset, limit, n_show):
    """How the final number was read against whether the answer was graded correct, and the endings of
    answers read without the marker. No GPU."""
    import torch
    _, mc = math_cfg()
    stem = "%s%s" % (dataset, "_pilot%d" % limit if limit else "")
    path = os.path.join(os.path.normpath(os.path.join(HERE, mc["out_dir"], model_folder)), "%s_sequences_v1.pt" % stem)
    d = torch.load(path, weights_only=False)
    how = np.array(d["extract_method"])
    wrong = np.array(d["all_hallucination_flag"]).astype(int)
    trunc = np.array(d["truncated"]).astype(bool)
    print("  %s" % path)
    print("  %-12s %7s %9s %10s" % ("read by", "answers", "correct", "truncated"))
    for m in ("marker", "last_number", "none"):
        s = how == m
        if s.any():
            print("  %-12s %7d %8.1f%% %10d" % (m, s.sum(), 100 * (1 - wrong[s].mean()), trunc[s].sum()))
    print("  truncated answers graded correct: %d of %d" % (int(((wrong == 0) & trunc).sum()), int(trunc.sum())))
    for label, sel in (("read by last number, graded CORRECT", (how == "last_number") & (wrong == 0) & ~trunc),
                       ("read by last number, graded WRONG", (how == "last_number") & (wrong == 1) & ~trunc)):
        idx = np.flatnonzero(sel)
        print("\n  %s: %d answers; endings of the first %d" % (label, len(idx), min(n_show, len(idx))))
        for i in idx[:n_show]:
            end = " ".join(d["decoded_text"][i][-170:].split())
            print("    read %s, reference %s | ...%s" % (d["pred"][i], d["gold"][i], end))


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("    [%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail))
        ok = ok and bool(cond)

    check("reference number from a GSM8K solution", reference_number("She sells 16 - 3 - 4 = 9.\n9 * 2 = 18\n#### 18") == 18.0)
    check("reference number with a thousands comma", reference_number("...\n#### 1,250") == 1250.0)
    n, how = read_final_number("First 5 + 7 = 12.\nThen 12 * 3 = 36.\n#### 36")
    check("final number from the marker", (n, how) == (36.0, "marker"), str((n, how)))
    n, how = read_final_number("#### 10\nWait, that is wrong. 5 * 3 = 15.\n#### $15.00")
    check("the LAST marker wins, with a dollar sign", (n, how) == (15.0, "marker"), str((n, how)))
    n, how = read_final_number("So the total is 1,234 apples.")
    check("no marker: last number in the text", (n, how) == (1234.0, "last_number"), str((n, how)))
    n, how = read_final_number("I am not sure.")
    check("no number at all", (n, how) == (None, "none"), str((n, how)))
    n, how = read_final_number("The answer is -3.5.\n#### -3.5")
    check("negative decimal", n == -3.5, str(n))
    check("18.0 equals 18", same_number(18.0, 18) and not same_number(18.5, 18) and not same_number(None, 18))
    check("a trailing period is not part of the number", to_number("42.") == 42.0)

    comp, ended = cut_at_stop([5, 6, 7, 99, 0, 0], {99})
    check("answer is cut before the stop token", comp == [5, 6, 7] and ended)
    comp, ended = cut_at_stop([5, 6, 7], {99})
    check("no stop token means truncated", comp == [5, 6, 7] and not ended)
    check("question seed is stable and differs by question", question_seed(0, 3) == question_seed(0, 3)
          and question_seed(0, 3) != question_seed(0, 4))

    class FakeTok:                      # '</s>' is an ordinary token here (id 7); only <|im_end|> is special
        eos_token_id = 9
        all_special_ids = [8, 9]
        added_tokens_encoder = {"<|endoftext|>": 8, "<|im_end|>": 9}

        def convert_tokens_to_ids(self, s):
            return {"</s>": 7, "<|endoftext|>": 8, "<|im_end|>": 9}.get(s)
    check("an ordinary text token is never a stop", stop_token_ids(FakeTok()) == {8, 9}, str(sorted(stop_token_ids(FakeTok()))))

    # 4 questions x 3 answers: all correct, all wrong, mixed (2 right), mixed (1 right)
    pid = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    wrong = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
    s = summarize(pid, wrong, [False] * 12, ["marker"] * 11 + ["none"], [50] * 12, 3)
    check("mixed questions counted", (s["questions_mixed"], s["questions_all_correct"], s["questions_all_wrong"]) == (2, 1, 1),
          str((s["questions_mixed"], s["questions_all_correct"], s["questions_all_wrong"])))
    check("wrong answers inside mixed questions", s["wrong_answers_in_mixed_questions"] == 3, str(s["wrong_answers_in_mixed_questions"]))
    check("histogram of correct answers per question", s["correct_per_question_histogram"] == [1, 1, 1, 1],
          str(s["correct_per_question_histogram"]))
    check("answer accuracy", abs(s["answer_accuracy"] - 6 / 12) < 1e-9, "%.3f" % s["answer_accuracy"])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prefetch", action="store_true", help="download the dataset into the cache and exit")
    ap.add_argument("--inspect", action="store_true", help="how answers were read vs graded, with examples; no GPU")
    ap.add_argument("--n-show", type=int, default=12)
    ap.add_argument("--model_folder")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--limit-questions", type=int, default=0, help="first N questions only (a pilot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.self_test:
        print("  self-test")
        raise SystemExit(0 if self_test() else 1)
    if a.prefetch:
        _, mc = math_cfg()
        for ds in mc["datasets"]:
            print("  %s: %d questions cached" % (ds["name"], len(load_questions(ds))))
        return
    if not a.model_folder:
        raise SystemExit("--model_folder is required (or --self-test / --prefetch)")
    if a.inspect:
        inspect(a.model_folder, a.dataset, a.limit_questions, a.n_show)
        return
    run(a.model_folder, a.dataset, a.limit_questions, a.seed, a.device)


if __name__ == "__main__":
    main()
