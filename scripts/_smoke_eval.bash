#!/usr/bin/env bash
# Smoke eval: score method, seed-2.0-pro judge, 8 new models' baseline (10 cases/domain).
# Waits for the generate driver to finish (marker in logs/smoke_generate.log), then runs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

# ---- wait for generate to complete ----
echo ">>> waiting for generate to finish..."
while ! grep -q ">>> ALL DONE" logs/smoke_generate.log 2>/dev/null; do
    sleep 20
done
echo ">>> generate done; starting eval"

CONFIG="gpt-5.4"
JUDGE="seed-2.0-pro"
METHOD="score"
LIMIT=10
CONC=8

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

n=${#DOMAINS[@]}; i=0
for d in "${DOMAINS[@]}"; do
    i=$((i+1))
    echo ">>> [eval $i/$n] domain=$d judge=$JUDGE method=$METHOD"
    uv run python scripts/run_evaluate.py \
        --domain "$d" --construction-profile "$CONFIG" \
        --judge "$JUDGE" --method "$METHOD" --limit "$LIMIT" \
        --concurrency "$CONC" --with-recall \
        2>&1 | grep -E "score.start|score.leaderboard|score.fail|Error|Traceback" || true
done
echo ">>> EVAL ALL DONE"
