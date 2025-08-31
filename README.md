# ICRL – Information-Guided Chain-of-Thought Reinforcement Loop

This repo contains our reference implementation of **ICRL** (Entropy-Minimisation variant) together with evaluation scripts and YAML-driven experiment management.

---

## 1  Quick start

$ bash scripts/download_model.sh

$ conda activate ICRL          

$ bash scripts/run_icrl.sh Qwen2.5-Math-7B AIME-TTT

Set ALLOW_OVERWRITE= True in run_icrl.py to overwrite hyperparameters 
主要调整参数
k:            64         # 每轮采样数量 

context_len:  8192

rounds:       5          # 轮数
entropy: True         #是否启用entropy 来挑选sample
entropy_k:    16       #计算entropy时采样数量 default 16
entropy_penalty : True  #是否在计算entropy启用penalty(如果一个答案不合法，entropy+= log(entropy_k)/entropy_k)
summary_length : 500 #summary 生成max token 数， 需要保证 summary_length*rounds <context_len (需要给final answer 留出足够空间)


