for Model in Qwen/Qwen2.5-32B Qwen/Qwen2.5-Math-72B; do
  export SUMMARY_LEN=500
  export CONTEXT_LEN=32768
  for k in 32 64; do
    for entropy_k in 16 32; do
      for rounds in 4 8 16 32; do
        export K=$k
        export ENTROPY_K=$entropy_k
        export ROUNDS=$rounds
        bash scripts/run_icrl.sh $Model AIME-TTT
      done
    done
  done
done
