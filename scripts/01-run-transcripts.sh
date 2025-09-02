#!/usr/bin/env bash

# ---- CONFIG ----
REPO_DIR="$HOME/The-Book-Of-Andy"
# REPO_URL="https://github.com/BitterDone/The-Book-Of-Andy.git"
RSS_URL="https://mfceoproject.libsyn.com/rss2"
PY_SCRIPT="$REPO_DIR/run-transcription.py"
VENV_DIR="$REPO_DIR/.venv"

# ---- RUN TRANSCRIPTION ----
echo "[*] Running transcription pipeline..."
python "$PY_SCRIPT" --rss "$RSS_URL" --repo "$REPO_DIR"
