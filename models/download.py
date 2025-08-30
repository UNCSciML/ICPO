# =============================================================
# File: /home/jovyan/ICRL/models/download.py
# =============================================================
"""Bulk‑download backbone checkpoints to a local directory.

Usage (shell):
    # download all default models to ./models/<model_name>
    python download.py

    # or only some models
    python download.py Qwen2.5-Math-1.5B Qwen2.5-7B

The script uses `huggingface_hub.snapshot_download`, so you need an
Internet connection and (for gated models) a valid HF access token in
`~/.huggingface/token` or the env var `HUGGINGFACE_HUB_TOKEN`.

If a target directory already exists it will be *skipped* (pass `--force`
to re‑download).
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download, HfApi, login

# ---------------------------------------------------------------------
# mapping from friendly name → Hub repo id
# ---------------------------------------------------------------------
MODEL_MAP: Dict[str, str] = {
    # Math base
    "Qwen2.5-Math-1.5B": "Qwen/Qwen2.5-Math-1.5B",
    "Qwen2.5-Math-7B": "Qwen/Qwen2.5-Math-7B",
    "Qwen2.5-Math-72B": "Qwen/Qwen2.5-Math-72B",
    # Vanilla base
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-32B": "Qwen/Qwen2.5-32B",
    # Instruct
    "LLaMA-3.1-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",  # gated
    "Qwen3-8B": "Qwen/Qwen3-8B",
    # Reasoning
    "Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
}

DEFAULT_MODELS = list(MODEL_MAP.keys())
SAVE_DIR = Path(__file__).resolve().parent  # /home/jovyan/ICRL/models

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def ensure_login():
    """Ensure we have a valid token for gated repos."""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()

    if token:          # 有 token 就静默登录一次
        login(token=token, add_to_git_credential=False)
    else:
        print("[WARN] No HF token found; gated models will fail to download.")



def download_one(name: str, repo: str, force: bool = False):
    tgt_dir = SAVE_DIR / name
    if tgt_dir.exists() and not force:
        print(f"[skip] {name} already exists → {tgt_dir}")
        return
    print(f"[↓] downloading {name} …")
    snapshot_download(
        repo_id=repo,
        cache_dir=tgt_dir,  # save straight into final dir
        local_dir=tgt_dir,
        resume_download=True,
        token=True,  # will trigger login if needed
        max_workers=8,
        tqdm_class=None,
    )
    # snapshot_download writes into local_dir; ensure pointer file removed
    print(f"[✓] saved to {tgt_dir}")

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS,
                        help="names to download (default: all presets)")
    parser.add_argument("--force", action="store_true", help="re‑download even if dir exists")
    args = parser.parse_args()

    ensure_login()

    for name in args.models:
        repo = MODEL_MAP.get(name)
        if not repo:
            print(f"[!] unknown model key: {name}. Available: {list(MODEL_MAP)}")
            continue
        download_one(name, repo, force=args.force)


if __name__ == "__main__":
    main()
