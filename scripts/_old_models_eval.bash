#!/usr/bin/env bash
# Score + Arena eval for 12 old models, gpt-5.4-high judge.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
JUDGE="gpt-5.4-high"
CONC=10
MODES="baseline"
MODELS="claude-opus-4.6-high,claude-sonnet-4.6-high,deepseek-v4-flash-high,gemini-3-flash-high,glm-5-thinking,glm-5.1-thinking,gpt-5.4-high,gpt-5.4-mini-high,kimi-k2.5-thinking,minimax-m2.5-thinking,minimax-m2.7-thinking,qwen-3.6-max-thinking"

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

echo ">>> OLD MODELS EVAL: score + arena, 12 models × 6 domains"

# Phase 1: Score
echo ">>> PHASE 1: score"
pids=()
for d in "${DOMAINS[@]}"; do
    (
        echo ">>> [score] domain=$d START"
        uv run python scripts/run_evaluate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --judge "$JUDGE" --method score \
            --models "$MODELS" --modes "$MODES" --concurrency "$CONC" \
            --with-recall \
            2>&1 | grep -E "score.start|score.leaderboard|score.fail|Error" | tail -5 || true
        echo ">>> [score] domain=$d DONE"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo ">>> PHASE 1 COMPLETE"

# Phase 2: Arena
echo ">>> PHASE 2: arena (20 pairs/case)"
pids=()
for d in "${DOMAINS[@]}"; do
    (
        echo ">>> [arena] domain=$d START"
        uv run python scripts/run_evaluate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --judge "$JUDGE" --method arena \
            --models "$MODELS" --modes "$MODES" --concurrency "$CONC" \
            --max-pairs-per-case 20 \
            2>&1 | grep -E "arena.start|arena.leaderboard|arena.fail|Error" | tail -5 || true
        echo ">>> [arena] domain=$d DONE"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo ">>> PHASE 2 COMPLETE"
echo ">>> OLD MODELS EVAL ALL DONE"
