#!/usr/bin/env bash
# Smoke a served MLA retrofit: coherence, thinking-off generation, and a long-ish prompt.
# Usage: smoke_mla.sh [base_url] [model_name]
set -euo pipefail
BASE=${1:-http://localhost:8000}
MODEL=${2:-qwen-mla}

ask() {  # ask <prompt> <max_tokens>
  curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- <<JSON
{"model": "$MODEL", "temperature": 0, "max_tokens": $2,
 "chat_template_kwargs": {"enable_thinking": false},
 "messages": [{"role": "user", "content": "$1"}]}
JSON
  echo
}

echo "== arithmetic (expect 391)"
ask "What is 17*23? Answer with the number only." 64
echo "== short factual (expect Paris)"
ask "The capital of France is" 64
echo "== echo, 40 words"
ask "Repeat exactly: alpha bravo charlie delta echo foxtrot golf hotel india juliet." 96
echo "== long-form sanity (expect coherent prose)"
ask "Write two sentences about why memory bandwidth limits attention decode." 200

# The context-merge path: a prefill that has cached context. It runs when a prompt is
# longer than --max-num-batched-tokens (chunk split) or when a prefix-cache hit reuses a
# block. Both are what QMLA-1 crashed on, so smoke them every time.
echo "== merge path: 1.2k-token prompt, cold then reused"
LONG=$(.venv/bin/python - <<'PY'
import json
body = ("The quick brown fox jumps over the lazy dog. " * 120) + \
    "\n\nRepeat the first five words above, then stop."
print(json.dumps({
    "model": "qwen-mla", "temperature": 0, "max_tokens": 24,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "user", "content": body}],
}))
PY
)
for arm in cold reused; do
  curl -sS -o /dev/null -w "  $arm: http=%{http_code} %{time_total}s\n" \
    "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$LONG"
done
curl -sS "$BASE/metrics" | grep -E '^vllm:prefix_cache_(queries|hits)_total' | sed 's/^/  /'
