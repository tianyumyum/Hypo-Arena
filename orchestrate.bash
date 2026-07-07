#!/usr/bin/env bash
# orchestrate.bash — full HypoArena pipeline launcher.
# Every CLI flag is listed below explicitly (even when it matches the default)
# so you can scan and confirm the active configuration before launch.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

# Raise process fd limit. macOS launchd defaults to soft=256, which 360+ async workers
# × 4 platform HTTPS clients exhaust quickly → cascading "APIConnectionError: Connection
# error." on every new socket. Kernel cap is kern.maxfilesperproc (typically 61440).
ulimit -n 16384 || true

# ---- Group A: scope (what to run) ----------------------------------------

DOMAINS="all"                            # comma list, or "all" for the 6 domains
CONSTRUCTION_PROFILE="gpt-5.4"           # reference forge profile (Responses API)
GEN_MODES="baseline,agent"               # subset of {baseline, agent}
WITH_RECALL_FLAG="--with-recall"         # use "--no-recall" to disable

# Judge profiles run IN PARALLEL — each produces its own
# `{config}.{judge}.{arena|score}.*` files. Add or remove entries freely.
JUDGES=(
    "mimo-v2-pro"
    "seed-2.0-pro"
)
JUDGES_CSV=$(IFS=,; echo "${JUDGES[*]}")

# 16 generation profiles, grouped by vendor (mirrors hypoarena/generate.bash PROFILES)
GEN_PROFILES=(
    "claude-sonnet-4.6-high"
    "claude-opus-4.6-high"
    "deepseek-v4-flash-high"
    "deepseek-v4-pro-high"
    "gemini-3-flash-high"
    "gemini-3.1-pro-high"
    "glm-5-thinking"
    "glm-5.1-thinking"
    "gpt-5.4-mini-high"
    "gpt-5.4-high"
    "kimi-k2.5-thinking"
    "kimi-k2.6-thinking"
    "minimax-m2.5-thinking"
    "minimax-m2.7-thinking"
    "qwen-3.6-max-thinking"
    "qwen-3.7-max-thinking"
)
# Joined form for CLI flag (do not edit directly; modify the array above).
GEN_PROFILES_CSV=$(IFS=,; echo "${GEN_PROFILES[*]}")

# ---- Group B: concurrency (workers per stage) ----------------------------

CONSTRUCTION_CONCURRENCY=4               # workers per domain (× 6 = 24)
GENERATION_CONCURRENCY=1                 # workers per (domain × mode × profile) (× 192 = 192)
ARENA_CONCURRENCY=16                     # workers per (domain × judge); 16 × 6 × 2 judges = 192
SCORE_CONCURRENCY=8                      # workers per (domain × judge); 8 × 6 × 2 judges = 96

# ---- Group C: robustness (retry, timeout) --------------------------------

MAX_ROUNDS=4                             # Forge–Audit max iterations per case
# (no retry cap; failed items retry forever with jittered backoff up to ~32 min)

# Per-task timeout (seconds). Catches hangs; transient errors are handled
# by FallbackModel + SDK retry layers below.
TASK_TIMEOUT_CONSTRUCTION=1800           # 30 min (biomedical with 4 Forge-Audit rounds + web_search runs ~14-16 min)
TASK_TIMEOUT_GENERATION=900              # 15 min
TASK_TIMEOUT_ARENA=300                   # 5 min
TASK_TIMEOUT_SCORE=300                   # 5 min

# ---- Group D: timing (polling) -------------------------------------------

POLL_INTERVAL=10                         # cross-stage scanner poll (seconds)
LEADERBOARD_INTERVAL=60                  # leaderboard rebuild cadence (seconds)

# ---- Group E: run mode ---------------------------------------------------

# DRY_RUN_FLAG: leave blank to actually run.
# Set to "--dry-run" to only report pending counts.
DRY_RUN_FLAG=""

# ---- Logging -------------------------------------------------------------

STAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="logs/orchestrate-${STAMP}.log"

# ---- Print resolved config ----------------------------------------------

cat <<INFO
HypoArena orchestrator launching at $(date '+%Y-%m-%d %H:%M:%S')
  domains                = ${DOMAINS}
  construction profile   = ${CONSTRUCTION_PROFILE}
  judges (${#JUDGES[@]})            = ${JUDGES_CSV}
  gen modes              = ${GEN_MODES}
  with recall            = ${WITH_RECALL_FLAG:-(disabled)}
  gen profiles (${#GEN_PROFILES[@]})      = ${GEN_PROFILES_CSV}

  concurrency:
    construction         = ${CONSTRUCTION_CONCURRENCY} (per domain → ×6 domains)
    generation           = ${GENERATION_CONCURRENCY} (per domain×mode×profile → ×192 triples)
    arena                = ${ARENA_CONCURRENCY} (per domain×judge → ×6×${#JUDGES[@]} = $((6 * ${#JUDGES[@]})) groups)
    score                = ${SCORE_CONCURRENCY} (per domain×judge → ×6×${#JUDGES[@]} = $((6 * ${#JUDGES[@]})) groups)

  robustness:
    max rounds (Forge)   = ${MAX_ROUNDS}
    retry policy         = infinite (jittered exp backoff, max ~32 min)
    task timeouts        = construction:${TASK_TIMEOUT_CONSTRUCTION}s
                           generation:${TASK_TIMEOUT_GENERATION}s
                           arena:${TASK_TIMEOUT_ARENA}s
                           score:${TASK_TIMEOUT_SCORE}s

  timing:
    poll interval        = ${POLL_INTERVAL}s
    leaderboard rebuild  = ${LEADERBOARD_INTERVAL}s

  mode:
    dry-run              = ${DRY_RUN_FLAG:-off}

  log file               = ${LOG_FILE}

  monitor: run `uv run python scripts/ops_monitor.py` in another terminal for live progress.
INFO

# ---- Launch --------------------------------------------------------------

uv run python scripts/run_orchestrate.py \
    --domains "${DOMAINS}" \
    --construction-profile "${CONSTRUCTION_PROFILE}" \
    --gen-profiles "${GEN_PROFILES_CSV}" \
    --gen-modes "${GEN_MODES}" \
    --judges "${JUDGES_CSV}" \
    ${WITH_RECALL_FLAG} \
    --construction-concurrency "${CONSTRUCTION_CONCURRENCY}" \
    --generation-concurrency "${GENERATION_CONCURRENCY}" \
    --arena-concurrency "${ARENA_CONCURRENCY}" \
    --score-concurrency "${SCORE_CONCURRENCY}" \
    --max-rounds "${MAX_ROUNDS}" \
    --task-timeout-construction "${TASK_TIMEOUT_CONSTRUCTION}" \
    --task-timeout-generation "${TASK_TIMEOUT_GENERATION}" \
    --task-timeout-arena "${TASK_TIMEOUT_ARENA}" \
    --task-timeout-score "${TASK_TIMEOUT_SCORE}" \
    --poll-interval "${POLL_INTERVAL}" \
    --leaderboard-interval "${LEADERBOARD_INTERVAL}" \
    ${DRY_RUN_FLAG} \
    2>&1 | tee "${LOG_FILE}"
