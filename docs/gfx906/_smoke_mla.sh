#!/usr/bin/env bash
# Smoke a served MLA retrofit: coherence, thinking-off generation, and a long-ish prompt.
# Usage: smoke_mla.sh [base_url] [model_name]
set -euo pipefail
BASE=${1:-http://localhost:8000}
MODEL=${2:-qwen-mla}

ask() {  # ask <prompt> <max_tokens>
  curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- <<JSON
{"model": "$MODEL", "temperature": 0, "max_tokens": $2,
 "messages": [{"role": "user", "content": "$1"}]}
JSON
  echo
}

echo "== arithmetic (expect 391)"
ask "What is 17*23? Answer with the number only." 32
echo "== short factual (expect Paris)"
ask "The capital of France is" 16
echo "== echo, 40 words"
ask "Repeat exactly: alpha bravo charlie delta echo foxtrot golf hotel india juliet." 60
echo "== long-form sanity (expect coherent prose)"
ask "Write two sentences about why memory bandwidth limits attention decode." 120
