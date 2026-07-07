#!/usr/bin/env bash
# Backfill: generate missing submissions + eval (score + arena) for all 13 models.
# Idempotent — existing records are skipped.
# Arena: all cases (no limit), 20 pairs/case, seed=42.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
JUDGE="gpt-5.4-high"
CONC=10

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

# Models with missing submissions (generate gap)
GEN_MODELS=(claude-opus-4.7-high claude-opus-4.8-high glm-5.2-thinking gpt-5.5-high mimo-v2.5-pro minimax-m3-thinking seed-2.1-pro)

# All 13 models for eval
ALL_MODELS="gpt-5.5-high,claude-opus-4.8-high,claude-opus-4.7-high,gemini-3.5-flash-high,gemini-3.1-pro-high,qwen-3.7-max-thinking,deepseek-v4-pro-high,kimi-k2.6-thinking,glm-5.2-thinking,seed-2.1-pro,mimo-v2.5-pro,minimax-m3-thinking,qwen-latest-beta-thinking"

# --- Phase 1: Generate missing submissions ---
echo ">>> PHASE 1: generate补跑 (${#GEN_MODELS[@]} models × ${#DOMAINS[@]} domains)"
pids=()
for model in "${GEN_MODELS[@]}"; do
    for d in "${DOMAINS[@]}"; do
        (
            echo ">>> [gen] domain=$d model=$model START"
            uv run python scripts/run_generate.py \
                --domain "$d" --construction-profile "$CONFIG" \
                --profile "$model" --mode baseline --concurrency "$CONC" \
                2>&1 | tail -3
            echo ">>> [gen] domain=$d model=$model DONE"
        ) &
        pids+=($!)
    done
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo ">>> PHASE 1 COMPLETE"

# --- Phase 2: Score eval (all 13 models, idempotent) ---
echo ">>> PHASE 2: score eval补跑 (13 models × 6 domains)"
pids=()
for d in "${DOMAINS[@]}"; do
    (
        echo ">>> [score] domain=$d START"
        uv run python scripts/run_evaluate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --judge "$JUDGE" --method score \
            --models "$ALL_MODELS" --modes baseline --concurrency "$CONC" \
            --with-recall \
            2>&1 | grep -E "score.start|score.leaderboard|score.fail|Error|Traceback" | tail -20 || true
        echo ">>> [score] domain=$d DONE"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo ">>> PHASE 2 COMPLETE"

# --- Phase 3: Arena eval (all 13 models, all cases, 20 pairs/case) ---
echo ">>> PHASE 3: arena eval补跑 (13 models × 6 domains, 20 pairs/case)"
pids=()
for d in "${DOMAINS[@]}"; do
    (
        echo ">>> [arena] domain=$d START"
        uv run python scripts/run_evaluate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --judge "$JUDGE" --method arena \
            --models "$ALL_MODELS" --modes baseline --concurrency "$CONC" \
            --max-pairs-per-case 20 \
            2>&1 | grep -E "arena.start|arena.leaderboard|arena.fail|Error|Traceback" | tail -20 || true
        echo ">>> [arena] domain=$d DONE"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo ">>> PHASE 3 COMPLETE"
echo ">>> BACKFILL ALL DONE"
