#!/usr/bin/env bash
# Trailing eval: score-only (rubric), full quantity, all passed cases.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
JUDGE="gpt-5.4-high"
CONC=16
MODES="baseline"
MODELS="gpt-5.5-high,claude-opus-4.8-high,claude-opus-4.7-high,gemini-3.5-flash-high,gemini-3.1-pro-high,qwen-3.7-max-thinking,deepseek-v4-pro-high,kimi-k2.6-thinking,glm-5.2-thinking,seed-2.1-pro,mimo-v2.5-pro,minimax-m3-thinking"

GENLOG="logs/full_generate.log"
DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

domain_started() { grep -q ">>> \[.*domain=$1 " "$GENLOG" 2>/dev/null; }
gen_all_done()   { grep -q ">>> FULL GENERATE ALL DONE" "$GENLOG" 2>/dev/null; }

n=${#DOMAINS[@]}
for ((k=0; k<n; k++)); do
    d="${DOMAINS[$k]}"
    nxt=""; (( k+1 < n )) && nxt="${DOMAINS[$((k+1))]}"
    echo ">>> [eval $((k+1))/$n] waiting for generate to finish domain=$d ..."
    while true; do
        if [ -n "$nxt" ]; then
            domain_started "$nxt" && break
        fi
        gen_all_done && break
        sleep 30
    done

    echo ">>> [eval $((k+1))/$n] domain=$d SCORE (full, judge=$JUDGE)"
    uv run python scripts/run_evaluate.py \
        --domain "$d" --construction-profile "$CONFIG" \
        --judge "$JUDGE" --method score \
        --models "$MODELS" --modes "$MODES" --concurrency "$CONC" \
        --with-recall \
        2>&1 | grep -E "score.start|score.leaderboard|score.fail|Error|Traceback" || true

    echo ">>> [eval $((k+1))/$n] domain=$d DONE"
done
echo ">>> FULL EVAL ALL DONE"
