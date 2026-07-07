#!/usr/bin/env bash
# qwen-beta agent generate (6 domains parallel) + eval (score+arena, agent mode)
# Compare qwen-latest-beta vs qwen-3.7-max in agent mode.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export UV_LINK_MODE=copy

CONFIG="gpt-5.4"
PROF="qwen-latest-beta-thinking"
JUDGE="gpt-5.4-high"
CONC=8
MODELS="qwen-latest-beta-thinking,qwen-3.7-max-thinking"

DOMAINS=(biomedical_science financial_analysis it_operations machine_learning safety_investigation social_science)

# --- Phase 1: Agent generate for qwen-beta, 6 domains parallel ---
echo ">>> PHASE 1: agent generate (6 domains parallel)"
pids=()
for d in "${DOMAINS[@]}"; do
    (
        echo ">>> [gen-agent] domain=$d START"
        uv run python scripts/run_generate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --profile "$PROF" --mode agent --concurrency "$CONC" \
            2>&1 | grep -E "generate.start|generate.fail|Error|Traceback" || true
        echo ">>> [gen-agent] domain=$d DONE"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo ">>> PHASE 1 COMPLETE: agent generate done"

# --- Phase 2: Score + Arena eval, agent mode, 6 domains parallel ---
echo ">>> PHASE 2: eval (score + arena, agent mode, 6 domains parallel)"
pids=()
for d in "${DOMAINS[@]}"; do
    (
        echo ">>> [score-agent] domain=$d START"
        uv run python scripts/run_evaluate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --judge "$JUDGE" --method score \
            --models "$MODELS" --modes agent --concurrency "$CONC" \
            --with-recall \
            2>&1 | grep -E "score.start|score.leaderboard|score.fail|Error|Traceback" || true
        echo ">>> [score-agent] domain=$d DONE"
    ) &
    pids+=($!)
    (
        echo ">>> [arena-agent] domain=$d START (limit=40, 20 pairs/case)"
        uv run python scripts/run_evaluate.py \
            --domain "$d" --construction-profile "$CONFIG" \
            --judge "$JUDGE" --method arena \
            --models "$MODELS" --modes agent --concurrency "$CONC" \
            --limit 40 --max-pairs-per-case 20 \
            2>&1 | grep -E "arena.start|arena.leaderboard|arena.fail|Error|Traceback" || true
        echo ">>> [arena-agent] domain=$d DONE"
    ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo ">>> PHASE 2 COMPLETE: agent eval done"

# --- Summary ---
echo ">>> SUMMARY (agent mode)"
uv run python -c "
import json, glob
from collections import defaultdict

scores = defaultdict(lambda: defaultdict(list))
for f in glob.glob('artifacts/*/results/gpt-5.4.gpt-5.4-high.score.jsonl'):
    domain = f.split('/')[1]
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            m = r.get('model', '')
            if 'qwen' in m and m.startswith('agent:'):
                scores[m][domain].append(r['overall_score'])

print()
print('=' * 80)
print('AGENT SCORE: qwen-latest-beta vs qwen-3.7-max (judge=gpt-5.4-high)')
print('=' * 80)
totals = defaultdict(list)
for model in sorted(scores):
    vals_all = []
    for domain in sorted(scores[model]):
        vals = scores[model][domain]
        vals_all.extend(vals)
    avg = sum(vals_all)/len(vals_all) if vals_all else 0
    totals[model] = vals_all
    print(f'  {model:<45} avg={avg:.3f} ({avg*20:.1f}%) n={len(vals_all)}')
print('=' * 80)
"
echo ">>> QWEN BETA AGENT FULL DONE"
