#!/usr/bin/env bash
# Agent generate for 12 target models (excluding beta), 6 domains.
# Idempotent — existing submissions are skipped.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
CONC=8

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

# Models needing agent generate
MODELS=(gpt-5.5-high claude-opus-4.8-high claude-opus-4.7-high gemini-3.5-flash-high glm-5.2-thinking seed-2.1-pro mimo-v2.5-pro minimax-m3-thinking)

echo ">>> AGENT GENERATE: ${#MODELS[@]} models × ${#DOMAINS[@]} domains"
pids=()
for model in "${MODELS[@]}"; do
    for d in "${DOMAINS[@]}"; do
        (
            echo ">>> [agent-gen] domain=$d model=$model START"
            uv run python scripts/run_generate.py \
                --domain "$d" --construction-profile "$CONFIG" \
                --profile "$model" --mode agent --concurrency "$CONC" \
                2>&1 | tail -3
            echo ">>> [agent-gen] domain=$d model=$model DONE"
        ) &
        pids+=($!)
    done
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo ">>> AGENT GENERATE ALL DONE"
