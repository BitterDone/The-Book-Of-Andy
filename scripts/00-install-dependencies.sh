#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG ----
REPO_DIR="$HOME/The-Book-Of-Andy"
# REPO_URL="https://github.com/BitterDone/The-Book-Of-Andy.git"
RSS_URL="https://mfceoproject.libsyn.com/rss2"
PY_SCRIPT="$REPO_DIR/run-transcription.py"
VENV_DIR="$REPO_DIR/.venv"

# ---- CHECK & INSTALL SYSTEM DEPENDENCIES ----
echo "[*] Checking system dependencies..."

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[*] Installing ffmpeg..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
fi

# if ! command -v git >/dev/null 2>&1; then
#     echo "[*] Installing git..."
#     sudo apt-get update && sudo apt-get install -y git
# fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "[!] Python3 not found. Please install manually."
    exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
    echo "[*] Installing pip3..."
    sudo apt-get update && sudo apt-get install -y python3-pip
fi

# # ---- UPDATE OR CLONE REPO ----
# # if [ ! -d "$REPO_DIR" ]; then
# #     echo "[*] Cloning repository..."
#     # git clone "$REPO_URL" "$REPO_DIR"
# # else
# echo "[*] Updating repository..."
# # cd "$REPO_DIR"
# git pull --rebase
# # fi

# ---- SETUP PYTHON VENV ----
# cd "$REPO_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "[*] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# ---- INSTALL PYTHON DEPENDENCIES ----
echo "[*] Installing/updating Python packages..."
"$VENV_DIR/bin/pip" install --upgrade \
    pip \
    git+https://github.com/openai/whisper.git \
    feedparser \
    requests \
    torch \
    pyannote.audio \
    huggingface_hub \
    faster-whisper \
    librosa \
    soundfile
