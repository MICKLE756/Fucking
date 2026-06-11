#!/usr/bin/env bash
# Run the full model + each single-component ablation across several seeds.
#
# Usage:
#   bash scripts/run_ablations.sh "0 1 2" 25
#       arg1 = space-separated seeds (default "0 1 2")
#       arg2 = epoch_size override   (default = value in config.yaml, i.e. unset)
#
# Logs are written to results/logs/<config>_seed<seed>.log. After it finishes,
# summarize with:
#   python scripts/aggregate_results.py results/logs
#
# NOTE: this trains real models and needs the dataset + a GPU. The default
# config already points at the data paths; edit src/config.yaml if needed.
set -u

SEEDS="${1:-0 1 2}"
EPOCHS="${2:-}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"

EPOCH_ARG=()
if [[ -n "$EPOCHS" ]]; then
  EPOCH_ARG=(--set "epoch_size=${EPOCHS}")
fi

# config_name -> extra --set flags ('full' = no override = all components on)
declare -A CONFIGS=(
  [full]=""
  [no_self_loop_fix]="--set use_self_loop_fix=no"
  [no_necessity]="--set use_necessity=no"
  [no_pos_prior]="--set use_pos_prior=no"
  [no_distillation]="--set use_distillation=no"
  [no_emotion_transition]="--set use_emotion_transition=no"
  [fusion_gated]="--set fusion_mode=gated"
)

for seed in $SEEDS; do
  for name in "${!CONFIGS[@]}"; do
    flags=${CONFIGS[$name]}
    log="$LOGDIR/${name}_seed${seed}.log"
    echo "=== [${name}] seed=${seed} -> ${log} ==="
    # shellcheck disable=SC2086
    python main.py --seed "$seed" "${EPOCH_ARG[@]}" $flags 2>&1 | tee "$log"
  done
done

echo "All runs done. Summarize with: python scripts/aggregate_results.py $LOGDIR"
