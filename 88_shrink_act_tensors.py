"""
88_shrink_act_tensors.py -- ACT-ViT's input at fewer token slots, derived exactly from the stored N=20 file.
=====================================================================================================
THE QUESTION (ICLR 2027 review, fgja). "I believe that ACT-ViT can also use fewer layers and even fewer
tokens. It would be useful to compare both methods when they use about the same amount of storage."

STORAGE PER ANSWER, in stored numbers (hidden size D):
    LOTUS           peak 9D + quantile range 18D + update range 16D  = 43 D
    ACT-ViT L8 N20  8 layer slots x 20 token slots                  = 160 D   (the paper's setting)
    ACT-ViT L8 N10                                                   =  80 D
    ACT-ViT L8 N5                                                    =  40 D   (matched: 0.93x LOTUS)

WHY THIS IS EXACT, NOT AN APPROXIMATION. ACT-ViT's preprocessing (59.act_pool, asserted bit-identical to
their code) zero-pads the token axis to N_MAX = 100 and max-pools non-overlapping blocks of 100 / N
tokens. At N = 20 a slot is the max over 5 tokens; at N = 5 it is the max over 20 tokens, i.e. over 4
consecutive N = 20 slots. A max of maxes over a partition is the max over the union, so

    act_pool(A, 8, 5) == max over groups of 4 consecutive slots of act_pool(A, 8, 20)

exactly, whenever 100 is divisible by both N and the source N is divisible by the target N. The
self-test checks this against act_pool itself on random tensors of every length up to 64 tokens. (The
layer axis does NOT shrink exactly this way for every model, because its padding depends on the
layer count, so fewer LAYER slots come from a fresh 59 extraction instead.)

The source file is read in chunks straight from the .npz (np.savez stores members uncompressed, so
the array is memory-mapped in place). TriviaQA's 114-130 GB source never has to fit in RAM.

  python 88_shrink_act_tensors.py --self-test
  python 88_shrink_act_tensors.py --dataset truthfulqa --model_folder qwen-2.5-7b-instruct --n-to 5
"""

import argparse
import json
import os
import struct
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AT_DIR = os.path.abspath(os.path.join(HERE, "..", "data-acttensors"))
N_MAX = 100          # ACT-ViT's zero-pad target (59.N_MAX)
CHUNK = 2048         # answers per chunk


def npz_member(path, member):
    """A read-only memmap of one array inside an uncompressed .npz, or None if it is compressed."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            return None
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        if local[:4] != b"PK\x03\x04":
            raise SystemExit("%s: bad zip local header for %s" % (path, member))
        name_len, extra_len = struct.unpack("<HH", local[26:30])
        f.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset,
                     order="F" if fortran else "C")


def shrink_tokens(at_chunk, n_to):
    """(B, L, N_src, D) -> (B, L, n_to, D), max over groups of N_src / n_to consecutive slots."""
    b, l, n_src, d = at_chunk.shape
    check_exact(n_src, n_to)
    return at_chunk.reshape(b, l, n_to, n_src // n_to, d).max(axis=3)


def check_exact(n_src, n_to):
    if N_MAX % n_src or N_MAX % n_to or n_src % n_to:
        raise SystemExit("not an exact re-pool: need %d %% N == 0 for both and %d %% %d == 0"
                         % (N_MAX, n_src, n_to))


def shrink_file(src, dst, n_to, chunk=CHUNK, log=print):
    t0 = time.time()
    at = npz_member(src, "at")
    if at is None:
        log("    %s is compressed; loading it whole" % os.path.basename(src))
        at = np.load(src)["at"]
    n, l, n_src, d = at.shape
    check_exact(n_src, n_to)
    out = np.empty((n, l, n_to, d), dtype=at.dtype)
    for a in range(0, n, chunk):
        out[a:a + chunk] = shrink_tokens(np.asarray(at[a:a + chunk]), n_to)
        if (a // chunk) % 10 == 0:
            log("    %d/%d answers (%.0fs)" % (min(a + chunk, n), n, time.time() - t0))
    src_gb = at.nbytes / 1024 ** 3
    del at                               # release the memmap before anything else opens the file
    with np.load(src) as z:
        rest = {k: z[k] for k in z.files if k != "at"}
    tmp = dst + ".partial.npz"           # np.savez appends .npz unless the name already ends in it
    np.savez(tmp, at=out, **rest)
    os.replace(tmp, dst)
    meta_src = src.replace(".npz", ".json")
    meta = {}
    if os.path.exists(meta_src):
        with open(meta_src) as f:
            meta = json.load(f)
    meta.update({"n_pool": int(n_to), "derived_from": os.path.abspath(src),
                 "derivation": ("exact token re-pool: max over groups of %d consecutive N=%d slots "
                                "(88_shrink_act_tensors.py); equal to act_pool(A, %d, %d) because "
                                "both pad to N_MAX=%d" % (n_src // n_to, n_src, l, n_to, N_MAX)),
                 "gb": round(out.nbytes / 1024 ** 3, 2), "seconds": round(time.time() - t0, 1)})
    with open(dst.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("  wrote %s  %s  %.2f GB (source %.2f GB)  in %.0fs"
        % (dst, out.shape, out.nbytes / 1024 ** 3, src_gb, time.time() - t0))
    return dst


def run(dataset, model_folder, at_dir, n_from, n_to, l_pool=8, force=False):
    src = os.path.join(at_dir, model_folder, "%s_at_L%d_N%d.npz" % (dataset, l_pool, n_from))
    dst = os.path.join(at_dir, model_folder, "%s_at_L%d_N%d.npz" % (dataset, l_pool, n_to))
    if not os.path.exists(src):
        raise SystemExit("%s not found -- extract it with 59_extract_act_tensors.py first" % src)
    if os.path.exists(dst) and not force:
        print("  SKIP: %s exists (--force to redo)" % dst)
        return dst
    print("  [%s/%s] L%d N%d -> N%d" % (model_folder, dataset, l_pool, n_from, n_to), flush=True)
    return shrink_file(src, dst, n_to, log=lambda s: print(s, flush=True))


# ---------------------------------------------------------------------------------------------

def self_test():
    print("=" * 78)
    print("  SELF-TEST: 88_shrink_act_tensors")
    print("=" * 78)
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("s59", os.path.join(HERE, "59_extract_act_tensors.py"))
    s59 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s59)
    assert s59.N_MAX == N_MAX, "59.N_MAX changed; the exactness argument assumes %d" % N_MAX
    rng = np.random.default_rng(0)

    # 1. Exactness against ACT-ViT's own preprocessing, for every answer length up to 64 tokens and
    # two layer counts (Qwen 29 hidden states, LLaMA 33).
    n_checked = 0
    for L1 in (29, 33):
        for T in list(range(1, 65)):
            A = rng.standard_normal((L1, T, 6)).astype(np.float32)
            p20 = s59.act_pool(A, 8, 20)
            for n_to in (10, 5):
                want = s59.act_pool(A, 8, n_to)
                got = shrink_tokens(p20[None], n_to)[0]
                assert np.array_equal(got, want), (L1, T, n_to)
                n_checked += 1
            # 59's default extraction is N=100 (one token per slot); its re-pool to 20 must be exact too.
            p100 = s59.act_pool(A, 8, 100)
            assert np.array_equal(shrink_tokens(p100[None], 20)[0], p20), (L1, T, "100->20")
            n_checked += 1
    print("  [PASS] re-pooled N=20 equals act_pool at N=10 and N=5, and N=100 re-pooled equals N=20, "
          "exactly (%d cases, 1-64 tokens, 29 and 33 layers)" % n_checked)

    # 2. Non-nested sizes are refused rather than approximated.
    for bad in ((20, 3), (20, 8), (30, 5)):
        try:
            check_exact(*bad)
            raise AssertionError("accepted a non-exact re-pool %r" % (bad,))
        except SystemExit:
            pass
    print("  [PASS] sizes that are not an exact re-pool are refused")

    # 3. The memmap reader returns the array np.load returns, and the whole file round-trips with
    # every other array untouched and chunk boundaries that do not divide the row count.
    with tempfile.TemporaryDirectory() as td:
        n = 37
        at = rng.standard_normal((n, 8, 20, 6)).astype(np.float16)
        at[3] = np.nan                                   # an empty completion, as 59 leaves it
        src = os.path.join(td, "x_at_L8_N20.npz")
        np.savez(src, at=at, prompt_id=np.arange(n) // 10, label=np.arange(n) % 2,
                 beam_row=np.arange(n)[::-1], token_len=np.full(n, 7))
        with open(src.replace(".npz", ".json"), "w") as f:
            json.dump({"n_pool": 20, "model_folder": "x"}, f)
        mm = npz_member(src, "at")
        assert mm is not None and np.array_equal(np.asarray(mm), at, equal_nan=True)
        del mm
        dst = os.path.join(td, "x_at_L8_N5.npz")
        shrink_file(src, dst, 5, chunk=8, log=lambda s: None)
        with np.load(dst) as z, np.load(src) as zs:
            got = z["at"]
            assert got.shape == (n, 8, 5, 6) and got.dtype == np.float16
            assert np.array_equal(got, shrink_tokens(at.astype(np.float32), 5).astype(np.float16),
                                  equal_nan=True)
            assert np.isnan(got[3]).all(), "an empty completion must stay NaN"
            for k in ("prompt_id", "label", "beam_row", "token_len"):
                assert np.array_equal(z[k], zs[k]), k
        with open(dst.replace(".npz", ".json")) as f:
            meta = json.load(f)
        assert meta["n_pool"] == 5 and meta["model_folder"] == "x" and "exact" in meta["derivation"]
        assert not os.path.exists(dst + ".partial.npz")
    print("  [PASS] memmap read, chunked re-pool and write: same keys, NaN rows kept, meta updated")

    print("\n  ALL PASS")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset", choices=["truthfulqa", "tydiqa_gp", "nq_open", "triviaqa"])
    p.add_argument("--model_folder")
    p.add_argument("--at-dir", default=DEFAULT_AT_DIR)
    p.add_argument("--n-from", type=int, default=20)
    p.add_argument("--n-to", type=int, default=5)
    p.add_argument("--l-pool", type=int, default=8)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not a.dataset or not a.model_folder:
        raise SystemExit("--dataset and --model_folder are required (or use --self-test)")
    run(a.dataset, a.model_folder, a.at_dir, a.n_from, a.n_to, a.l_pool, a.force)


if __name__ == "__main__":
    main()
