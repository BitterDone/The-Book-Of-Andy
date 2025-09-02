#!/usr/bin/env bash

# ---- COMMIT & PUSH RESULTS ----
git add transcripts/*.txt || true

if ! git diff-index --quiet HEAD; then
    git commit -m "Add new transcripts $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    git push
else
    echo "[*] No new transcripts to commit."
fi
