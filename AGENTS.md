# opencode Project Preferences — HOSVD Hallucination Detection

## NEVER DO (without explicit permission)
- Push code unless told "push now" / "push code" / "push"
- Commit unless told to commit
- Write code when asked for analysis/answers only
- Assume anything — ask if unclear
- Use non-ASCII characters in Python files (em-dash, arrows, box-drawing)
- Cache intermediate tensors (U_L, U_D) to disk — always recompute
- Add new features/code when only asked a question

## ALWAYS DO
- Run `git pull` before any cluster execution
- Activate conda env `hal-det` on cluster
- Use `config.yaml` for model/dataset config (not hardcoded)
- Follow HARP's evaluation protocol: known/unknown split, beam search, contrastive judge
- Set `TF_FORCE_GPU_ALLOW_GROWTH=true` before any TensorFlow import
- Set `HF_METRICS_CACHE=/tmp/rouge_cache_<JOB_ID>` for concurrent SLURM jobs
- Strip non-ASCII chars from Python files

## DATA CONVENTIONS
- Pooled tensors: `../data/{model_folder}/{dataset}_pooled.pt` or `{suffix}.pt`
- Raw/unpooled: `../data_unpooled/{model_folder}/`
- SLURM output: `slurm_{dataset}_{%j}.out`
- Model folders match config.yaml: `llama-3.1-8b-instruct`, `qwen-2.5-7b-instruct`
- File suffix conventions: `_fullbeams`, `_maxenergy`, `_fullbeams`
- `.gitignore` blocks `*.pt`, `*.pth`, `*.safetensors`, `data_unpooled/`

## CLUSTER
- Path: `/home/de807845/Hallucination-Detection/hallucination-detection`
- SLURM template: `-p highgpu --gres=gpu:1 --mem=80G`
- Conda: `module load anaconda/anaconda-2024.10 cuda/cuda-12.6.0 && conda activate hal-det`
- Do NOT `source /share/apps/anaconda3-2022.05/...` -- that path does not exist on Newton. A
  `source` of it under `set -e` kills the job in under a second, so sbatch returns a job id for a
  job that never starts and squeue shows nothing. Cost one submission on 2026-09-08.
- In .slurm files use `set -u`, not `set -euo pipefail`, and put `|| exit 1` on each real command
- `mkdir -p slurm_logs` belongs in the submit script, on the login node. SLURM does not create the
  directory for `--output`, and a job whose log cannot be opened dies before it runs.
- EVERY .slurm file must strip inherited conda state before `module load`:
      for _v in $(compgen -v | grep '^CONDA' || true); do unset "$_v"; done
      unset _CE_M _CE_CONDA || true
  sbatch exports the submitting shell's environment, so submitting from a `(hal-det)`
  prompt hands the job a conda stack it cannot unwind. Lmod's anaconda module retries
  `conda deactivate` forever: 16,958 identical CondaErrors, then death on the wall clock.
  Cost four jobs on 2026-09-08 and was misdiagnosed twice as other things. It is
  INTERMITTENT -- it depends only on whether the submitter had an env active, so the same
  script appears to work and then hang. `slurm/hgreb_stage.slurm` has the same shape and
  the same latent bug.
- Working reference: `slurm/hgreb_stage.slurm`. Copy from it, not from this section.
- NEVER `pip install` into `hal-det` without checking what it drags in. `pip install
  vit-pytorch` pulled `torchvision==0.28.0` from PyPI, whose build does not match the env's
  `torch 2.13.0+cu126`. Every transformers model import then died with `RuntimeError:
  operator torchvision::nms does not exist` -- the whole env, not just the new code. Cost
  four jobs on 2026-09-08. `pip download` first and read the tree.
- transformers tolerates torchvision being ABSENT; it breaks on one that is present and
  ABI-mismatched. So `pip uninstall torchvision` restores transformers, and `59`/`42`/`39`
  work again immediately -- none of them need it.
- `vit-pytorch` DOES need torchvision: its `__init__.py` imports `dino`, which imports
  `torchvision.transforms`. `--no-deps` alone is therefore not enough. Install a build that
  matches the env's torch:
      pip install --no-deps torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
  and verify BOTH before submitting anything:
      python -c "from transformers import AutoModelForCausalLM; from vit_pytorch import ViT"
  (`from vit_pytorch.vit import ViT` would bypass `dino`, but ACT-ViT's own code uses the
  package-level import, and their files stay unmodified.)
- ROUGE race condition fix: isolate `HF_METRICS_CACHE` per job
- Clean `/tmp` after runs: `rm -rf /tmp/rouge_cache_*`

## BLEURT/ROUGE
- BLEURT-20 benchmark via `evaluate.load("bleurt", config_name="BLEURT-20")`
- ROUGE-L threshold: 0.7
- BLEURT threshold: 0.5
- Batch predictions against references (repeat prediction N times)
- `TF_FORCE_GPU_ALLOW_GROWTH=true` prevents TensorFlow seizing GPU VRAM

## EVALUATION PROTOCOL
- No system prompt — use raw dataset prompt templates
- 10-beam search per prompt (HARP-compatible)
- Known = any beam correct; Unknown = all beams wrong
- Split: 75% known → train, 25% known + ALL unknown → valid
- Contrastive judge: BLEURT vs correct AND incorrect answers
- Full-beam protocol for main results; early-stop was deprecated

## HOSVD
- Optimal: R_L=5, R_D=64, offset=0 (grid search confirmed)
- Reasoning window: layers 15-23 (9 layers) — dynamic EGTE used gradient-based positioning
- Mean-pooling deprecated → max-energy pooling preferred
- No intermediate U_L/U_D caching
- RF is baseline classifier; LR/MLP for ablation
- Zero leakage: factor matrices from train only
- Gram matrix trick (eigh, not SVD) with chunked matmul for memory
- HARP-HOSVD fusion attempted but lm_head SVD proved destructive to covariance structure

## CLASSIFIERS
- Best: RF on 4/7 datasets, LR on 2/7, MLP on 1/7
- No single classifier dominates — RF is safest default
- StandardScaler helps LR/MLP, not RF
- 5k subset for quick tests; full train for final numbers

## FILE NAMING
- Auto-increment: `{ID:02d}_description.py` (e.g., 01_generate, 02_hosvd_evaluate)
- Unused/legacy scripts deleted, not left to rot

## KNOWN BUGS & FIXES
- PyYAML rejects non-ASCII characters even in comments → strip all
- ROUGE batch mismatch → repeat prediction per reference
- BLEURT batch mismatch → same fix
- cuSOLVER thread-safety → multiprocessing with spawn context
