#!/usr/bin/env bash
# Recovery: regenerate gemini-3.5-flash-high baseline submissions for all 6 domains
# (they were deleted by an over-broad glob). Idempotent — only this one model.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"; MODE="baseline"; PROF="gemini-3.5-flash-high"; CONC=10
DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

for d in "${DOMAINS[@]}"; do
    echo ">>> regen domain=$d profile=$PROF"
    uv run python scripts/run_generate.py \
        --domain "$d" --construction-profile "$CONFIG" \
        --profile "$PROF" --mode "$MODE" --concurrency "$CONC" \
        2>&1 | grep -E "generate.start|generate.fail|Error|Traceback" || true
done
echo ">>> GEMINI REGEN DONE"
