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
import argparse, datetime,json, csv, random, re, logging
from pathlib import Path
from typing import List, Dict, Any,Union
import math



import torch, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from vllm import LLM,SamplingParams
from icrl.prompt_cache import PromptCache

# ---------------- Config & Debug ----------------
DEBUG = True      # ⇦ 如无需日志，改为 False
MAX_KEEP = 5      # 每类历史最多保留条数
TENSOR_PARALLEL = torch.cuda.device_count()


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

ENABLE_PENALTY = None
def debug_jsonl(filename: Union[str, Path], data: dict):
    """
    将单个 dict 写入 JSONL 文件，每行一个 JSON 对象。
    如果需要写多个 dict，可以先循环调用本函数。

    Args:
        filename: 输出文件路径
        data: 单个 dict
    """
    out_file = Path(filename)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")

def _extract_numeric(solution_str:str, method="strict")->str:
    assert method in ["strict", "flexible"]
    #from https://github.com/PRIME-RL/TTRL/blob/main/verl/verl/utils/reward_score/gsm8k.py, modify the search patteren into boxed{}
    if method == "strict":
        # match \boxed{number}
        solution = re.findall(r"\\boxed\{(\-?[0-9\.\,]+)\}", solution_str)
        if not solution: 
            final_answer = None
        else:
            final_answer = solution[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def _normalize(ans: str) -> float:
    raw = ans
    if re.fullmatch(r"-?\d+(\.\d+)?", ans.strip()):  # if input is a str of number
        ans = ans.strip()
    else:
        ans = _extract_numeric(ans)
    if ans is None:
        return None
    ans = ans.replace("$", "").replace(",", "").replace(" ", "")
    if ans.startswith("+"):
        ans = ans[1:]
    if ans.endswith("."):
        ans = ans[:-1]
    
    try:
        ans_float = float(ans)
    except ValueError:
        return None
    
    if DEBUG:
        logging.debug(f"[normalize] '{raw[:40]}' -> {ans_float}")
    
    return ans_float


def _get_entropy(lst: List[str], max_classes: int) -> float | None:
    """
    计算给定字符串列表的归一化答案熵。
    对无效答案按比例加权惩罚，惩罚权重基于最大类别数对应的最大熵。

    Args:
        lst: 候选答案列表
        max_classes: 预估最大类别数，用于计算最大熵作为惩罚基数
        enable_penalty: 是否启用无效样本惩罚

    Returns:
        float: 熵值（含惩罚）
        None: 如果列表为空或无有效答案
    """
    global ENABLE_PENALTY
    total_count = len(lst)
    if total_count == 0:
        return None

    cnt = {}
    valid_count = 0
    penalty_weight = math.log2(max_classes)  # 最大可能的熵，惩罚基数

    for a in lst:
        norm_a = _normalize(a)
        if not norm_a:
            continue
        cnt[norm_a] = cnt.get(norm_a, 0) + 1
        valid_count += 1

    if valid_count == 0:
        # 全无效样本，返回较大惩罚值
        return penalty_weight * 5

    entropy = -sum((v / valid_count) * math.log2(v / valid_count) for v in cnt.values())
    if entropy==0 and DEBUG:
            logging.info(f"[eval] Entropy=0 ,values = {cnt}")
            logging.info(f"[eval] Entropy=0 ,lst = {lst}")
    if ENABLE_PENALTY:

        invalid_ratio = (total_count - valid_count) / total_count
        if DEBUG:
            logging.info(f"[eval] ENABLE_PENALTY,invalid_rato = {invalid_ratio}")
        entropy += penalty_weight * invalid_ratio

    return entropy
def strip_reasoning(txt: str) -> str:
    if "</think>" in txt.lower():
        txt = txt.lower().split("</think>", 1)[-1]
    return txt.strip()

# ---------- prompt builders ----------

SYS_PROMPT = """
You are an AI mathematician, and you must always respond in English.

Your answers must be clear explanations in plain English only.  
- Do NOT include code, pseudocode, equations, or technical snippets.  
- Focus on step-by-step reasoning and concept clarification.  

You will also be given compressed solution ideas from earlier questions.  
These are NOT answers to the current question.  
They are only examples of reasoning style to guide how you explain your solution.  

For the current question, you must combine its details with the reasoning style of the examples to deduce the correct numeric answer.

After finishing all reasoning, output your final answer on a new line, in the exact format:

boxed{<number>}

where <number> is the final numeric result only (a decimal number, not a fraction or expression).  
Do not include any other text in the final output.
"""
def select_samples(lines,model,tok,args):
    sys_prompt=lines[0]
    current_QA=lines[-1]
    samples=lines[1:-1]
    entropy_ls = []
    for idx,sample in enumerate(samples):
        prompt_ls = [sys_prompt,sample,current_QA]
        prompt = '\n'.join(prompt_ls).rstrip()
        result= generate_batch(model =model,
                       tok=tok,
                       prompts=[prompt],
                       n=args.entropy_k,
                       max_new=args.answer_length,
                       temp=args.temp,
                       top_p=args.top_p
                       )[0]
        entropy = _get_entropy(result,args.entropy_k)
        entropy_ls.append((idx,entropy))
    entropy_sorted = sorted(entropy_ls, key=lambda x: x[1])
    new_lines = [sys_prompt]
    print("entropy_sorted",entropy_sorted)
    for i in  range(args.max_num):
        idx,entropy= entropy_sorted[i]
        new_lines.append(samples[idx])
    new_lines.append(current_QA)
    return new_lines


    

def build_prompt_from_training(cur_q: str, hist: List[dict],cur_idx =None,answer = None,model=None,tok=None,args=None) -> str:
    lines = [SYS_PROMPT.rstrip()+"\nPrevious questions and answers:\n"]
    if DEBUG:
            logging.info(f"[eval] cur_idx = {cur_idx}")
    if hist !=[]:
        for i,h in enumerate(hist):
            idx =h['idx']
            sum=h['sum']
            q=h['question']
            # label =h['label']
            if idx!=cur_idx:
                print(f"cur_idx:{cur_idx},idx:{idx}")
                lines.append(f"previous question [{i}]:{q}\n answer of [{i}]:{sum}\n")
    else:
        lines.append("None.")
    max_num=args.max_num
    if answer is None:
        lines.append(f"current question:{cur_q}\ncurrent answer:")
    else:
        prompt=" Your task is to generate a concise reasoning path that leads naturally to this label, and then provide the final answer.\n"
        prompt = f"current question:{cur_q}\n Ground truth label: {answer}\n current answer:"
        lines.append(prompt)
    if max_num!=None and len(lines)-2>max_num: # len(lines)= num(samples)+sys_prompt+question
        lines = select_samples(lines,model,tok,args)
        
    return "\n".join(lines).rstrip()



# ---------- LLM wrappers ----------
def _gen(model, tok, prompt: str, max_new: int,
         temp: float, top_p: float) -> str:
    # enc = tok(prompt, return_tensors="pt").to(model.device)
    # with torch.no_grad():
    #     out = model.generate(
    #         **enc,
    #         max_new_tokens=max_new,
    #         min_new_tokens=8,
    #         do_sample=True,
    #         temperature=temp,
    #         top_p=top_p,
    #         repetition_penalty=1.1,
    #         pad_token_id=tok.eos_token_id
    #     )[0]
    # return tok.decode(out[enc["input_ids"].shape[1]:],
    #                   skip_special_tokens=True).strip()
    sampling_params = SamplingParams(
        n=1,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_new,
        min_tokens=8,
        repetition_penalty=1.1,
       stop_token_ids=[tok.eos_token_id], 
    )
    outputs = model.generate([prompt], sampling_params)
   
    return outputs[0].outputs[0].text.strip()


def generate_batch(model, tok, prompts: List[str], n: int, max_new: int,
                   temp: float, top_p: float) -> List[List[str]]:
    if DEBUG and prompts:
        logging.info(f"[generate_batch] first prompt len={len(prompts[0])}")
    # enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    # base_lens = enc["attention_mask"].sum(dim=1)
    # with torch.no_grad():
    #     out = model.generate(
    #         **enc,
    #         num_return_sequences=n,
    #         max_new_tokens=max_new,
    #         min_new_tokens=8,
    #         do_sample=True,
    #         temperature=temp,
    #         top_p=top_p,
    #         repetition_penalty=1.1,
    #         pad_token_id=tok.eos_token_id
    #     )
    # out = out.view(len(prompts), n, -1).cpu()
    # res: List[List[str]] = []
    # for i in range(len(prompts)):
    #     base_len = int(base_lens[i])
    #     gens = [strip_reasoning(tok.decode(out[i, j, base_len:], skip_special_tokens=True))
    #             for j in range(n)]
    #     res.append(gens)
    #     if DEBUG and i == 0:
    #         logging.info(f"[generate_batch] sample gens={gens[:3]}")
    # return res
    
    sampling_params = SamplingParams(
        n=n,                       # 等价于 num_return_sequences
        max_tokens=max_new,        # 新生成 token 数
        min_tokens=8,
        temperature=temp,
        top_p=top_p,
        repetition_penalty=1.1,
        
        stop_token_ids=[tok.eos_token_id],  # 防止跑太长
    )

    
    outputs = model.generate(prompts, sampling_params)

    res: List[List[str]] = []
    for i, out in enumerate(outputs):
        gens = [
            strip_reasoning(o.text) 
            for o in out.outputs
        ]
        res.append(gens)
        if DEBUG and i == 0:
            logging.info(f"[generate_batch] sample gens={gens[:3]}")

    return res

# ---------- summarisation ----------
_SUMMARY_PROMPT = (
    "You will be given a question, an attempted reasoning process, and the ground truth answer, "
    "which appears in the format label [<answer>]. "
    "Your task is to provide a clear, concise summary of the reasoning that leads to the answer. "
    "Do not include any numerical calculations. "
    "The summary must be self-contained and no longer than 100 tokens. \n\n"
    "At the end, include the final numeric result in the format boxed{{<number>}}, "
    "where <number> is a decimal (no fractions, variables, or extra text). \n\n"
    "[Answer start]\n{}\n[Answer end]\n\n"
    "Summary:"
)


def _fallback_summary(ans_raw: str) -> str:
    first = _SENT_SPLIT.split(ans_raw.strip())[0]
    first = re.sub(r"\$.*?\$", " ", first)
    first = re.sub(r"[^A-Za-z\s]", " ", first).strip()
    return first or "reasoning unavailable"

def _compress_train_answer(ans_raw: str, model, tok,output_dir=None) -> str:
    if not ans_raw.strip():
        return ""
    prompt = _SUMMARY_PROMPT.format(ans_raw.strip())
    summary = _gen(model, tok, prompt, max_new=500, temp=0.05, top_p=0.9).strip()
    summary = summary
    if output_dir:
        debug_jsonl(output_dir/"debug.jsonl",{"ans_raw":ans_raw,"summary":summary})
    return summary

# ---------- metric ----------
def mean_at_k(preds, refs):
    return sum(sum(_normalize(c) == _normalize(r) for c in cand)/len(cand)
               for cand, r in zip(preds, refs)) / len(preds)


     
def set_global_variable(args):
    global ENABLE_PENALTY
    ENABLE_PENALTY=args.enable_penalty
            
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task_dir",   required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--csv_path",   default=f"{ROOT_DIR}/results/icrl_new_idea/metrics.csv")
    ap.add_argument("--k",      type=int,   default=cfg("k", 16))
    ap.add_argument("--batch",  type=int,   default=cfg("batch", 4))
    ap.add_argument("--temp",   type=float, default=cfg("temperature", 0.6))
    ap.add_argument("--top_p",  type=float, default=cfg("top_p", 0.95))
    ap.add_argument("--ctx",    type=int,   default=cfg("context_len", 3072))
   
 
    ap.add_argument("--entropy_k",type = int,default= cfg("entropy_k", 16))
    ap.add_argument("--enable_penalty",type =bool,default =True)
    ap.add_argument("--answer_length",type = int,default =5000)
    ap.add_argument("--train_sample",type = int,default =10)
    ap.add_argument("--test_sample",type = int,default =None)
    ap.add_argument("--max_num",type = int,default =5)
    ap.add_argument_group
    args = ap.parse_args()

    set_global_variable(args)

    
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(f"{args.output_dir}_new_idea_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args)
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
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

    # tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path, torch_dtype=torch.float16,
    #     device_map="balanced", trust_remote_code=True).eval()

  

    model = LLM( # 用VLLM 因为其运行速度更快，支持prefix caching等
        model=args.model_path,
        trust_remote_code=True,
        tokenizer_mode="auto",    
        tensor_parallel_size=TENSOR_PARALLEL,  
        enable_prefix_caching=True,  
        enforce_eager=True          
    )
    tok = model.get_tokenizer()
    # ---------- load dataset ----------
    ds_path = Path(args.task_dir)
    file = "test.parquet" if (ds_path / "test.parquet").exists() else "test.json"
    dataset = load_dataset(
        "parquet" if file.endswith(".parquet") else "json",
        data_files=str(ds_path / file), split="train"
    )
    if args.test_sample is not None:
        dataset=dataset.select(range(args.test_sample))
    train_file = "train.parquet" if (ds_path / "train.parquet").exists() else "train.json"
    train_dataset = load_dataset(
        "parquet" if train_file.endswith(".parquet") else "json",
        data_files=str(ds_path / train_file), split="train"
    )
    train_dataset = train_dataset.select(range(args.train_sample))
  
    preds, refs, ans_records = [], [], []
    wrote_example = False

    pbar = tqdm(range(0, len(train_dataset), args.batch), unit="batch", desc="ICRL‑0‑train")
    train_history = []
    for st in pbar:
        ed  = min(st + args.batch, len(train_dataset))
        sub = train_dataset.select(range(st, ed))
        qs  = [ex.get("prompt") or ex["problem"] for ex in sub]
        refs_batch = [ex.get("answer") or ex.get("solution") or "" for ex in sub]
     
        prompts = []
        for q,ref in zip(qs,refs_batch):
            ans = _normalize(ref)
            prompt= build_prompt_from_training(q,train_history,answer= ans,model=model,tok=tok,args=args)
            prompts.append(prompt)
        max_new = max(32, args.ctx - max(len(tok(p).input_ids) for p in prompts))
        k_batch = generate_batch(model, tok, prompts,1, #generate answer for each input
                                     max_new, args.temp, args.top_p)
        for idx, (q, raw_ans, ref) in enumerate(zip(qs,k_batch,refs_batch)):
            ran_ans_with_label= f"question:{q}\n answer:"+raw_ans[0]+f"\nlabel: [{ref}] "
            summary =_compress_train_answer(ran_ans_with_label,model,tok,out_dir)
            summary+=f"Thus, the answer is boxed{ {_normalize(ref)} }"
            if summary:
                train_history.append({"question":q,'idx':st+idx,'label':_normalize(ref),'sum':summary})

    pbar = tqdm(range(0, len(dataset), args.batch), unit="batch", desc="ICRL‑0‑m")
    if DEBUG:
        logging.debug(f"[train_history] {[d['idx'] for d in train_history]}")
    preds, refs, ans_records = [], [], []
    for st in pbar:
        ed  = min(st + args.batch, len(dataset))
        sub = dataset.select(range(st, ed))
        qs  = [ex.get("prompt") or ex["problem"] for ex in sub]
        refs_batch = [ex.get("answer") or ex.get("solution") or "" for ex in sub]
        refs+=refs_batch
        # ---------- evaluation ----------

        final_prompts = [build_prompt_from_training(q, train_history,i+st,answer=None,model=model,tok=tok,args=args) for i,q in enumerate(qs)]
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


        #----- final output and answer for debugging--------
        final_out_file = out_dir / "prompts_and_preds.jsonl"
        with open(final_out_file, "a", encoding="utf-8") as f:
            for prompt, gens in zip(final_prompts, final_k):
                # 每条写入一行 JSON
                json_line = json.dumps({"prompt": prompt, "gens": gens}, ensure_ascii=False)
                f.write(json_line + "\n")
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
              "mean@k": mean_k, "k": args.k,
              "batch": args.batch, "temp": args.temp, "top_p": args.top_p,
              "context_len": args.ctx,
              "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    (out_dir / "metrics.json").write_text(json.dumps(metric, indent=2),
                                          encoding="utf-8")

    csv_path = Path(args.csv_path); csv_path.parent.mkdir(parents=True, exist_ok=True)
    head = not csv_path.exists()
    print("csv_path",csv_path)
    with csv_path.open("a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(metric.keys()))
        if head: w.writeheader()
        w.writerow(metric)

    print(f"\n[✓] {metric['model']} on {metric['task']} → mean@k {metric['mean@k']}%")

if __name__ == "__main__":
    main()
