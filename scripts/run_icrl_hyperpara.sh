for Model in Qwen/Qwen2.5-32B Qwen/Qwen2.5-Math-72B; do
  for k in 32 64; do
    for entropy_k in 16 32; do
      for rounds in 4 8 16 32; do
        ENV_SUMMARY_LEN=500 \
        ENV_CONTEXT_LEN=32768 \
        ENV_K=$k \
        ENV_ENTROPY_K=$entropy_k \
        ENV_ROUNDS=$rounds \
        bash scripts/run_icrl.sh $Model AIME-TTT
      done
    done
  done
done