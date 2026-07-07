#!/usr/bin/env bash
# Full generate: 12 target models × 6 domains × ALL passed cases (no --limit).
# Idempotent — run_generate.py skips cases already present in each submission file,
# so the 4 pre-existing models (gemini-3.1-pro/qwen-3.7-max/deepseek-v4-pro/kimi-k2.6)
# only backfill the it_operations gap; the 8 new models fill everything.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
MODE="baseline"
CONC=10

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)
PROFILES=(
    gpt-5.5-high
    claude-opus-4.8-high
    claude-opus-4.7-high
    gemini-3.5-flash-high
    gemini-3.1-pro-high
    qwen-3.7-max-thinking
    deepseek-v4-pro-high
    kimi-k2.6-thinking
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
            --profile "$p" --mode "$MODE" --concurrency "$CONC" \
            2>&1 | grep -E "generate.start|generate.fail|Error|Traceback" || true
    done
done
echo ">>> FULL GENERATE ALL DONE"
