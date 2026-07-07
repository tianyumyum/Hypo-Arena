#!/usr/bin/env bash
# Smoke arena: seed-2.0-pro judge, limit 10, 5 domains (it_operations has 0 passed cases).
# Idempotent — existing old-model pairs are skipped; only new-model pairings run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
JUDGE="seed-2.0-pro"
METHOD="arena"
LIMIT=10
CONC=8

DOMAINS=(biomedical_science financial_analysis machine_learning safety_investigation social_science)

n=${#DOMAINS[@]}; i=0
for d in "${DOMAINS[@]}"; do
    i=$((i+1))
    echo ">>> [arena $i/$n] domain=$d judge=$JUDGE"
    uv run python scripts/run_evaluate.py \
        --domain "$d" --construction-profile "$CONFIG" \
        --judge "$JUDGE" --method "$METHOD" --limit "$LIMIT" \
        --concurrency "$CONC" \
        2>&1 | grep -E "arena.start|arena.leaderboard|arena.fail|Error|Traceback" || true
done
echo ">>> ARENA ALL DONE"
