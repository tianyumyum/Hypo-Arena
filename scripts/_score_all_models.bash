#!/usr/bin/env bash
# Score eval: all 13 models (12 targets + qwen-beta), gpt-5.4-high judge, 6 domains parallel.
# Idempotent — existing score records are skipped.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
JUDGE="gpt-5.4-high"
CONC=10
MODES="baseline"
MODELS="gpt-5.5-high,claude-opus-4.8-high,claude-opus-4.7-high,gemini-3.5-flash-high,gemini-3.1-pro-high,qwen-3.7-max-thinking,deepseek-v4-pro-high,kimi-k2.6-thinking,glm-5.2-thinking,seed-2.1-pro,mimo-v2.5-pro,minimax-m3-thinking,qwen-latest-beta-thinking"

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

echo ">>> SCORE ALL: 13 models, 6 domains parallel, judge=$JUDGE"
pids=()
for d in "${DOMAINS[@]}"; do
    (
        echo ">>> [score] domain=$d START"
        uv run python scripts/run_evaluate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --judge "$JUDGE" --method score \
            --models "$MODELS" --modes "$MODES" --concurrency "$CONC" \
            --with-recall \
            2>&1 | grep -E "score.start|score.leaderboard|score.fail|Error|Traceback" || true
        echo ">>> [score] domain=$d DONE"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo ">>> SCORE ALL DONE"
