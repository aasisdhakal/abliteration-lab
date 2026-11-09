#!/usr/bin/env bash
# Resumable benchmark run with the LM Evaluation Harness.
#
#   ./bench.sh <name> <model path or HF repo id>
#   ./bench.sh original Qwen/Qwen2.5-1.5B-Instruct
#   ./bench.sh heretic  aasis001/Qwen2.5-1.5B-Instruct-heretic-ara
#
# - Skips a suite whose results JSON already exists under bench/<name>/.
# - --use_cache stores every answered request in bench/cache/, so an aborted
#   suite continues from where it stopped instead of from 0%.
set -u
cd "$(dirname "$0")"
name=${1:?usage: ./bench.sh <name> <model>}
model=${2:?usage: ./bench.sh <name> <model>}
args="pretrained=$model,dtype=bfloat16"
mkdir -p bench/cache

done_already() {  # $1=task: true if a results JSON for that task exists
  # find, not a glob: lm_eval saves local models under hidden ".__<name>" folders
  find "bench/$name" -name 'results_*.json' -exec grep -lq "\"$1\"" {} + 2>/dev/null
}

if done_already mmlu; then
  echo "=== $name: multiple-choice suite (already done, skipping) ==="
else
  echo "=== $name: multiple-choice suite ==="
  lm_eval --model hf --model_args "$args" --batch_size auto \
      --tasks mmlu,arc_challenge,hellaswag,winogrande,piqa \
      --use_cache "bench/cache/$name-mc" \
      --output_path "bench/$name" > "bench/$name-mc.log" 2>&1 || echo "FAILED $name mc"
fi

if done_already gsm8k; then
  echo "=== $name: gsm8k (already done, skipping) ==="
else
  echo "=== $name: gsm8k (first 250) ==="
  lm_eval --model hf --model_args "$args" --batch_size auto \
      --tasks gsm8k --limit 250 \
      --use_cache "bench/cache/$name-gsm8k" \
      --output_path "bench/$name" > "bench/$name-gsm8k.log" 2>&1 || echo "FAILED $name gsm8k"
fi
echo "DONE"
