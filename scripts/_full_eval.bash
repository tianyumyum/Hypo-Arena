#!/usr/bin/env bash
# Full eval: gemini-3.5-flash-high judge (native json_schema, no parse failures),
# arena + score, restricted to the 12 target models (+reference), ALL passed cases.
# Waits for the full generate driver to finish first.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

# ---- wait for generate to complete ----
echo ">>> waiting for full generate to finish..."
while ! grep -q ">>> FULL GENERATE ALL DONE" logs/full_generate.log 2>/dev/null; do
    sleep 30
done
echo ">>> generate done; starting eval"

CONFIG="gpt-5.4"
JUDGE="gemini-3.5-flash-high"
METHOD="both"
CONC=10
MODELS="gpt-5.5-high,claude-opus-4.8-high,claude-opus-4.7-high,gemini-3.5-flash-high,gemini-3.1-pro-high,qwen-3.7-max-thinking,deepseek-v4-pro-high,kimi-k2.6-thinking,glm-5.2-thinking,seed-2.1-pro,mimo-v2.5-pro,minimax-m3-thinking"

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

n=${#DOMAINS[@]}; i=0
for d in "${DOMAINS[@]}"; do
    i=$((i+1))
    echo ">>> [eval $i/$n] domain=$d judge=$JUDGE method=$METHOD"
    uv run python scripts/run_evaluate.py \
        --domain "$d" --construction-profile "$CONFIG" \
        --judge "$JUDGE" --method "$METHOD" \
        --models "$MODELS" --concurrency "$CONC" --with-recall \
        2>&1 | grep -E "arena.start|arena.leaderboard|arena.fail|score.start|score.leaderboard|score.fail|Error|Traceback" || true
done
echo ">>> FULL EVAL ALL DONE"
