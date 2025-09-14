#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update

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
    sudo apt-get install -y ffmpeg
fi

# if ! command -v git >/dev/null 2>&1; then
#     echo "[*] Installing git..."
#     sudo apt-get install -y git
# fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "[!] Python3 not found. Please install manually."
    exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
    echo "[*] Installing pip3..."
    sudo apt-get install -y python3-pip
fi

# Check if libcudnn is installed
if ! ls /usr/lib/x86_64-linux-gnu/libcudnn* &>/dev/null; then
    echo "[*] cuDNN not found. Installing libcudnn8 and libcudnn8-dev..."
    sudo apt-get install -y libcudnn8 libcudnn8-dev
else
    echo "[*] cuDNN already installed."
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
    feedparser \
    requests \
    torch \
    pyannote.audio \
    huggingface_hub \
    faster-whisper \
    librosa \
    soundfile \
    huggingface_hub \
    ffmpeg \
    git+https://github.com/m-bain/whisperx.git
    # git+https://github.com/openai/whisper.git \ # removed for whisperx for better timestamps
