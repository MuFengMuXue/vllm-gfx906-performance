#!/usr/bin/env bash
# Diagnose a served model's first-token distribution: near-uniform means the hidden state is
# garbage; a peaked-but-wrong token means wrong values (weights/rope/layout).
set -euo pipefail
BASE=${1:-http://localhost:8000}
MODEL=${2:-qwen-mla}

echo "== next-token distribution for a fixed prompt (top 5)"
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- <<JSON | "$(dirname "$0")/../../.venv/bin/python" -c '
import json,sys
r=json.load(sys.stdin)
ch=r["choices"][0]
for k,v in (ch["logprobs"]["content"][0]["top_logprobs"] if ch.get("logprobs") else []):
    print(f"  {v:8.3f}  {k!r}")
print("chosen:", repr(ch["message"]["content"]))
' 2>/dev/null || true
{"model": "$MODEL", "temperature": 0, "max_tokens": 1, "logprobs": true, "top_logprobs": 5,
 "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}]}
JSON

echo "== raw completion (no chat template)"
curl -s "$BASE/v1/completions" -H 'Content-Type: application/json' -d @- <<JSON
{"model": "$MODEL", "temperature": 0, "max_tokens": 8, "prompt": "The capital of France is"}
JSON
echo
