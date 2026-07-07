#!/usr/bin/env bash
# Patch: re-run baseline generate + score for models with gaps.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
JUDGE="gpt-5.4-high"
CONC=10

# Models with baseline gaps
PATCH_MODELS=(claude-opus-4.8-high claude-opus-4.7-high glm-5.2-thinking minimax-m3-thinking)
DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)
ALL_MODELS="gpt-5.5-high,claude-opus-4.8-high,claude-opus-4.7-high,gemini-3.5-flash-high,gemini-3.1-pro-high,qwen-3.7-max-thinking,deepseek-v4-pro-high,kimi-k2.6-thinking,glm-5.2-thinking,seed-2.1-pro,mimo-v2.5-pro,minimax-m3-thinking,qwen-latest-beta-thinking"

echo ">>> BASELINE PATCH: generate"
pids=()
for model in "${PATCH_MODELS[@]}"; do
    for d in "${DOMAINS[@]}"; do
        (
            uv run python scripts/run_generate.py \
                --domain "$d" --construction-profile "$CONFIG" \
                --profile "$model" --mode baseline --concurrency "$CONC" \
                2>&1 | tail -2
        ) &
        pids+=($!)
    done
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo ">>> BASELINE PATCH: generate DONE"

echo ">>> BASELINE PATCH: score"
pids=()
for d in "${DOMAINS[@]}"; do
    (
        uv run python scripts/run_evaluate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --judge "$JUDGE" --method score \
            --models "$ALL_MODELS" --modes baseline --concurrency "$CONC" \
            --with-recall \
            2>&1 | grep -E "score.start|score.leaderboard|score.fail" | tail -5 || true
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo ">>> BASELINE PATCH: score DONE"
echo ">>> BASELINE PATCH ALL DONE"
