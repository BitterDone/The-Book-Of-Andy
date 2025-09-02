#!/usr/bin/env bash

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

# ---- CONFIG ----
REPO_DIR="$HOME/The-Book-Of-Andy"
# REPO_URL="https://github.com/BitterDone/The-Book-Of-Andy.git"
RSS_URL="https://mfceoproject.libsyn.com/rss2"
PY_SCRIPT="$REPO_DIR/run-transcription.py"
VENV_DIR="$REPO_DIR/.venv"

# ---- RUN TRANSCRIPTION ----
echo "[*] Running transcription pipeline..."
python "$PY_SCRIPT" --rss "$RSS_URL" --repo "$REPO_DIR"
