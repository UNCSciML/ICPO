# =============================================================
# File: src/icrl/icrl0_runner.py          (rev 2025‑07‑06‑m)
# =============================================================
"""
ICRL‑0 – compressed, English‑only  (稳定版 + DEBUG).

修复日志
--------
1. 摘要为空不写入历史，并截断历史长度 (MAX_KEEP=5)。
2. 用 Markdown 列表代替 `{ ... }` 区块，避免模型复制花括号。
3. generation 统一加 repetition_penalty=1.1。
4. _extract_numeric 先从最后一行反扫只含数字/分数的行。
5. 保留 DEBUG 日志；DEBUG=False 时完全关闭。
"""

from __future__ import annotations
import argparse, datetime, json, csv, random, re, logging
from pathlib import Path
from typing import List, Dict, Any

import torch, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from icrl.prompt_cache import PromptCache

# ---------------- Config & Debug ----------------
DEBUG = True      # ⇦ 如无需日志，改为 False
MAX_KEEP = 5      # 每类历史最多保留条数

ROOT_DIR   = Path(__file__).resolve().parents[2]
CONFIG_YML = ROOT_DIR / "configs" / "icrl.yaml"
_cfg: Dict[str, Any] = yaml.safe_load(CONFIG_YML.open()) if CONFIG_YML.is_file() else {}
def cfg(k, d): return _cfg.get(k, d)

# ---------- regex ----------
_THINK_TAGS = re.compile(r"</?think>", re.I)
_BOXED      = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.I)
_NUMBER     = re.compile(r"-?\d+(?:/\d+)?(?:\.\d+)?")
_LETTER     = re.compile(r"\b([A-E])\b", re.I)
_SENT_SPLIT = re.compile(r"[.!?]")

# ---------- helpers ----------
def _extract_numeric(txt: str) -> str:
    """
    优先从文本 **最后一行**开始，寻找只包含数字/分数/小数的片段；
    若找不到，再回退到原逻辑 (boxed / 最后出现数字 / 单选 A‑E)。
    """
    if not txt:
        return ""
    # strip think tags
    txt = _THINK_TAGS.sub(" ", txt)
    # 从后往前遍历行
    for ln in reversed(txt.strip().splitlines()):
        ln = ln.strip()
        if re.fullmatch(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", ln):
            return ln
    # boxed {...}
    if (m := _BOXED.search(txt)):
        return m.group(1)
    # 最后一个数字
    nums = _NUMBER.findall(txt.replace(",", ""))
    if nums:
        return nums[-1]
    # 选项字母
    lets = _LETTER.findall(txt)
    if lets:
        return lets[-1]
    return txt.strip()

def _normalize(ans: str) -> str:
    raw = ans
    ans = _extract_numeric(ans)
    ans = ans.replace("$", "").replace(",", "").replace(" ", "")
    if ans.startswith("+"): ans = ans[1:]
    if ans.endswith("."):   ans = ans[:-1]
    if re.fullmatch(r"-?\d+\.\d+", ans) and float(ans).is_integer():
        ans = str(int(float(ans)))
    if re.fullmatch(r"-?\d+", ans):
        ans = str(int(ans))
    if DEBUG:
        logging.debug(f"[normalize] '{raw[:40]}' -> '{ans}'")
    return ans.lower()

def _majority_raw(lst: List[str]) -> str:
    cnt, raw = {}, {}
    for a in lst:
        k = _normalize(a)
        if not k:
            continue
        cnt[k] = cnt.get(k, 0) + 1
        raw.setdefault(k, a)
    if not cnt:
        return random.choice(lst)
    top = max(cnt.values())
    winners = [k for k, v in cnt.items() if v == top]
    choice = random.choice(winners)
    return raw[choice]

def strip_reasoning(txt: str) -> str:
    if "</think>" in txt.lower():
        txt = txt.lower().split("</think>", 1)[-1]
    return txt.strip()

# ---------- prompt builders ----------
SYS_PROMPT = (
    "You are an AI mathematician. All content you output MUST be in English.\n"
    "Below are compressed solution ideas from previous attempts; each idea is tagged "
    "with reward 1 (correct) or reward 0 (incorrect). Use the question and these ideas "
    "to deduce the correct numeric answer. Do NOT output code or non‑English text.\n"
    "**Finish all your reasoning, then on a NEW line output exactly one number "
    "(the answer) and nothing else.**\n"
)

def build_prompt(q: str, hist: Dict[str, List[str]]) -> str:
    lines = [SYS_PROMPT.rstrip(), "", q.rstrip(), ""]
    lines.append("bad ideas (reward 0):")
    lines += [f"- {s}" for s in hist["bad"]]
    lines.append("")
    lines.append("good ideas (reward 1):")
    lines += [f"- {s}" for s in hist["good"]]
    return "\n".join(lines).rstrip()

# ---------- update history ----------
def update_history(h: Dict[str, List[str]], summary: str, reward: bool):
    """写入摘要到对应桶，并截断长度。summary 已保证非空。"""
    bucket = h["good"] if reward else h["bad"]
    bucket.append(summary)
    del bucket[:-MAX_KEEP]        # 只保留最后 MAX_KEEP 条

# ---------- LLM wrappers ----------
def _gen(model, tok, prompt: str, max_new: int,
         temp: float, top_p: float) -> str:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            min_new_tokens=8,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id
        )[0]
    return tok.decode(out[enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()

def generate_batch(model, tok, prompts: List[str], n: int, max_new: int,
                   temp: float, top_p: float) -> List[List[str]]:
    if DEBUG and prompts:
        logging.info(f"[generate_batch] first prompt len={len(prompts[0])}")
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    base_lens = enc["attention_mask"].sum(dim=1)
    with torch.no_grad():
        out = model.generate(
            **enc,
            num_return_sequences=n,
            max_new_tokens=max_new,
            min_new_tokens=8,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id
        )
    out = out.view(len(prompts), n, -1).cpu()
    res: List[List[str]] = []
    for i in range(len(prompts)):
        base_len = int(base_lens[i])
        gens = [strip_reasoning(tok.decode(out[i, j, base_len:], skip_special_tokens=True))
                for j in range(n)]
        res.append(gens)
        if DEBUG and i == 0:
            logging.info(f"[generate_batch] sample gens={gens[:3]}")
    return res

# ---------- summarisation ----------
_SUMMARY_PROMPT = (
    "Summarise the core reasoning of the following answer. "
    "Omit numerical calculations.\n\n[Answer start]\n{}\n[Answer end]"
)

def _fallback_summary(ans_raw: str) -> str:
    first = _SENT_SPLIT.split(ans_raw.strip())[0]
    first = re.sub(r"\$.*?\$", " ", first)
    first = re.sub(r"[^A-Za-z\s]", " ", first).strip()
    return first or "reasoning unavailable"

def _compress_answer(ans_raw: str, model, tok) -> str:
    if not ans_raw.strip():
        return ""
    prompt = _SUMMARY_PROMPT.format(ans_raw.strip())
    summary = _gen(model, tok, prompt, max_new=48, temp=0.05, top_p=0.9).strip()
    return summary if summary else _fallback_summary(ans_raw)

# ---------- metric ----------
def mean_at_k(preds, refs):
    return sum(sum(_normalize(c) == _normalize(r) for c in cand)/len(cand)
               for cand, r in zip(preds, refs)) / len(preds)

# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task_dir",   required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--csv_path",   default=f"{ROOT_DIR}/results/icrl0/metrics.csv")
    ap.add_argument("--k",      type=int,   default=cfg("k", 16))
    ap.add_argument("--batch",  type=int,   default=cfg("batch", 4))
    ap.add_argument("--temp",   type=float, default=cfg("temperature", 0.6))
    ap.add_argument("--top_p",  type=float, default=cfg("top_p", 0.95))
    ap.add_argument("--ctx",    type=int,   default=cfg("context_len", 3072))
    ap.add_argument("--rounds", type=int,   default=cfg("rounds", 3))
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- logging ----
    if DEBUG:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(out_dir / "debug.log", "w", "utf-8"),
                logging.StreamHandler()
            ]
        )
        logging.info("DEBUG MODE ON")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True).eval()

    # ---------- load dataset ----------
    ds_path = Path(args.task_dir)
    file = "test.parquet" if (ds_path / "test.parquet").exists() else "test.json"
    dataset = load_dataset(
        "parquet" if file.endswith(".parquet") else "json",
        data_files=str(ds_path / file), split="train"
    )

    cache = PromptCache(out_dir / ".prompt_cache")
    preds, refs, ans_records = [], [], []
    wrote_example = False

    pbar = tqdm(range(0, len(dataset), args.batch), unit="batch", desc="ICRL‑0‑m")

    for st in pbar:
        ed  = min(st + args.batch, len(dataset))
        sub = dataset.select(range(st, ed))
        qs  = [ex.get("prompt") or ex["problem"] for ex in sub]
        refs_batch = [ex.get("answer") or ex.get("solution") or "" for ex in sub]
        refs += refs_batch

        hist = [{"good": [], "bad": []} for _ in sub]

        # ---------- bootstrap ----------
        for r in range(1, args.rounds + 1):
            prompts = []
            for q, h in zip(qs, hist):
                key = f"{hash(q)}-r{r}-g{len(h['good'])}-b{len(h['bad'])}"
                p   = cache.get(key) or build_prompt(q, h)
                cache.put(key, p); prompts.append(p)

            max_new = max(32, args.ctx - max(len(tok(p).input_ids) for p in prompts))
            k_batch = generate_batch(model, tok, prompts, args.k,
                                     max_new, args.temp, args.top_p)
            pseudo  = [_majority_raw(lst) for lst in k_batch]

            rand_idx = [random.randrange(args.k) for _ in prompts]
            picked   = [lst[i] for lst, i in zip(k_batch, rand_idx)]

            for h, ans_raw, pseudo_raw in zip(hist, picked, pseudo):
                reward  = _normalize(ans_raw) and _normalize(ans_raw) == _normalize(pseudo_raw)
                summary = _compress_answer(ans_raw, model, tok).strip()
                if summary:                       # 只记录非空摘要
                    update_history(h, summary, reward)

        # ---------- evaluation ----------
        final_prompts = [build_prompt(q, h) for q, h in zip(qs, hist)]
        max_new = max(32, args.ctx - max(len(tok(p).input_ids) for p in final_prompts))
        final_k = generate_batch(model, tok, final_prompts, args.k,
                                 max_new, args.temp, args.top_p)
        preds += final_k

        if DEBUG:
            logging.info(f"[eval] normalized k list sample = {[_normalize(x) for x in final_k[0]]}")

        if not wrote_example:
            (out_dir / "example.txt").write_text(
                f"PROMPT:\n{final_prompts[0]}\n\nANSWER:\n{final_k[0][0]}",
                encoding="utf-8"
            )
            wrote_example = True

        # --------- record answers ---------
        for idx, (klist, ex, ref_raw) in enumerate(zip(final_k, sub, refs_batch)):
            ref_norm = _normalize(ref_raw)
            cand_list = [{"raw": a,
                          "numeric": _normalize(a),
                          "correct": bool(_normalize(a) and _normalize(a) == ref_norm)}
                         for a in klist]
            ans_records.append({
                "id": ex.get("id", st + idx),
                "ref": ref_norm,
                "candidates": cand_list,
                "normalized_candidates": [c["numeric"] for c in cand_list]
            })

    # ---------- write outputs ----------
    mean_k = round(mean_at_k(preds, refs) * 100, 2)
    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps({"gens": g}, ensure_ascii=False) for g in preds),
        encoding="utf-8"
    )
    (out_dir / "answer.json").write_text(
        json.dumps(ans_records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    metric = {"model": Path(args.model_path).name, "task": ds_path.name,
              "mean@k": mean_k, "k": args.k, "rounds": args.rounds,
              "batch": args.batch, "temp": args.temp, "top_p": args.top_p,
              "context_len": args.ctx,
              "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    (out_dir / "metrics.json").write_text(json.dumps(metric, indent=2),
                                          encoding="utf-8")

    csv_path = Path(args.csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
    head = not csv_path.exists()
    with csv_path.open("a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(metric.keys()))
        if head: w.writeheader()
        w.writerow(metric)

    print(f"\n[✓] {metric['model']} on {metric['task']} → mean@k {metric['mean@k']}%")

if __name__ == "__main__":
    main()