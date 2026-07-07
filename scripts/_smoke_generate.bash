#!/usr/bin/env bash
# Smoke generate: 8 new models × 6 domains × 10 baseline cases each.
# Restartable — run_generate.py skips cases already present in each submission file.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
MODE="baseline"
LIMIT=10
CONC=10

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)
PROFILES=(
    gpt-5.5-high
    claude-opus-4.8-high
    claude-opus-4.7-high
    gemini-3.5-flash-high
    glm-5.2-thinking
    seed-2.1-pro
    mimo-v2.5-pro
    minimax-m3-thinking
)

total=$(( ${#DOMAINS[@]} * ${#PROFILES[@]} ))
i=0
for d in "${DOMAINS[@]}"; do
    for p in "${PROFILES[@]}"; do
        i=$((i+1))
        echo ">>> [$i/$total] domain=$d profile=$p"
        uv run python scripts/run_generate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --profile "$p" --mode "$MODE" --limit "$LIMIT" --concurrency "$CONC" \
            2>&1 | grep -E "generate.start|generate.fail|Error|Traceback" || true
    done
done
echo ">>> ALL DONE"
