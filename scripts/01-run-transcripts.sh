#!/usr/bin/env bash

# ---- CHECK HUGGING FACE AUTH ----
echo "[*] Checking Hugging Face authentication..."
echo "Do you need to accept T/C to access a gated HF model?"
# https://huggingface.co/pyannote/segmentation-3.0
# https://huggingface.co/pyannote/speaker-diarization-3.1

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

# ---- CHECK HUGGING FACE MODEL ACCESS ----
echo "[*] Verifying access to pyannote/speaker-diarization-3.1..."

if ! huggingface-cli whoami &>/dev/null; then
    echo "[!] Hugging Face CLI not authenticated."
    echo "    Run: huggingface-cli login"
    exit 1
fi

# Try to check access by hitting the model page API
ACCESS_CHECK=$(curl -s -H "Authorization: Bearer ${HF_TOKEN:-}" https://huggingface.co/api/models/pyannote/speaker-diarization-3.1)

if echo "$ACCESS_CHECK" | grep -q '"private":true'; then
    if echo "$ACCESS_CHECK" | grep -q '"gated":true'; then
        echo "[!] The model 'pyannote/speaker-diarization-3.1' is gated."
        echo "    Please visit: https://huggingface.co/pyannote/speaker-diarization-3.1"
        echo "    and accept the terms of use with your Hugging Face account."
        exit 1
    fi
fi

if echo "$ACCESS_CHECK" | grep -q '"error"'; then
    echo "[!] Could not verify access to pyannote/speaker-diarization-3.1."
    echo "    Make sure your HF_TOKEN is set and valid."
    echo "    Run: export HF_TOKEN=your_token_here"
    exit 1
fi

echo "[✓] Hugging Face access to diarization model confirmed."

# ---- CONFIG ----
REPO_DIR="$HOME/The-Book-Of-Andy"
# REPO_URL="https://github.com/BitterDone/The-Book-Of-Andy.git"
RSS_URL="https://mfceoproject.libsyn.com/rss2"
PY_SCRIPT="$REPO_DIR/run-transcription.py"
VENV_DIR="$REPO_DIR/.venv"

# ---- RUN TRANSCRIPTION ----
echo "[*] Running transcription pipeline..."
"$VENV_DIR/bin/python" "$PY_SCRIPT" --rss "$RSS_URL" --repo "$REPO_DIR" --token "$HF_TOKEN"
