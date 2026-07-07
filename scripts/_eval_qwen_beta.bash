#!/usr/bin/env bash
# Wait for qwen-beta generate to finish, then score eval across all 6 domains.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
JUDGE="gemini-3.5-flash-medium"
PROF="qwen-latest-beta-thinking"
CONC=16
MODES="baseline"

GENLOG="logs/gen_qwen_beta.log"
DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

echo ">>> waiting for qwen-beta generate to finish..."
while ! grep -q ">>> QWEN BETA GENERATE DONE" "$GENLOG" 2>/dev/null; do
    sleep 30
done
echo ">>> qwen-beta generate done; starting score eval"

for d in "${DOMAINS[@]}"; do
    echo ">>> [eval] domain=$d model=$PROF SCORE (judge=$JUDGE)"
    uv run python scripts/run_evaluate.py \
        --domain "$d" --construction-profile "$CONFIG" \
        --judge "$JUDGE" --method score \
        --models "$PROF" --modes "$MODES" --concurrency "$CONC" \
        --with-recall \
        2>&1 | grep -E "score.start|score.leaderboard|score.fail|Error|Traceback" || true
    echo ">>> [eval] domain=$d DONE"
done
echo ">>> QWEN BETA EVAL ALL DONE"
