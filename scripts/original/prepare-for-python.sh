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

if ! command -v git >/dev/null 2>&1; then
    echo "[*] Installing git..."
    sudo apt-get update && sudo apt-get install -y git
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "[!] Python3 not found. Please install manually."
    exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
    echo "[*] Installing pip3..."
    sudo apt-get update && sudo apt-get install -y python3-pip
fi

# ---- UPDATE OR CLONE REPO ----
# if [ ! -d "$REPO_DIR" ]; then
#     echo "[*] Cloning repository..."
    # git clone "$REPO_URL" "$REPO_DIR"
# else
echo "[*] Updating repository..."
# cd "$REPO_DIR"
git pull --rebase
# fi

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
pip install --upgrade \
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
    
# ---- CHECK HUGGING FACE AUTH ----
echo "[*] Checking Hugging Face authentication..."

HF_TOKEN_FILE="$HOME/.huggingface/token"

check_hf_auth() {
    if [ -n "${HF_TOKEN:-}" ]; then
        return 0
    elif [ -f "$HF_TOKEN_FILE" ] && [ -s "$HF_TOKEN_FILE" ]; then
        return 0
    else
        return 1
    fi
}

if ! check_hf_auth; then
    echo "[!] No Hugging Face authentication found."
    echo "You need to authenticate before running diarization."
    echo "Options:"
    echo "  1. Run: huggingface-cli login"
    echo "  2. Or paste a token below (from https://huggingface.co/settings/tokens)."
    echo

    read -p "Paste Hugging Face token (or press Enter to skip): " user_token

    if [ -n "$user_token" ]; then
        mkdir -p "$(dirname "$HF_TOKEN_FILE")"
        echo "$user_token" > "$HF_TOKEN_FILE"
        chmod 600 "$HF_TOKEN_FILE"
        export HF_TOKEN="$user_token"
        echo "[✓] Token saved to $HF_TOKEN_FILE"
    else
        echo "[!] Authentication required. Exiting."
        deactivate
        exit 1
    fi
else
    echo "[✓] Hugging Face authentication found."
fi

# ---- RUN TRANSCRIPTION ----
echo "[*] Running transcription pipeline..."
python "$PY_SCRIPT" --rss "$RSS_URL" --repo "$REPO_DIR"

# ---- COMMIT & PUSH RESULTS ----
git add transcripts/*.txt || true

if ! git diff-index --quiet HEAD; then
    git commit -m "Add new transcripts $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    git push
else
    echo "[*] No new transcripts to commit."
fi

# ---- DEACTIVATE VENV ----
deactivate
