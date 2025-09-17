#!/usr/bin/env bash
set -euo pipefail

# Resolve Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
# https://deeptalk.lambdalabs.com/t/how-to-install-libcudnn-8-ubuntu-20-04-nvidia-smi-460-56-driver-version-460-56-cuda-version-11-2/3138
# https://stackoverflow.com/questions/66977227/could-not-load-dynamic-library-libcudnn-so-8-when-running-tensorflow-on-ubun
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
export last_public_key=3bf863cc
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/${last_public_key}.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8
sudo apt-get install libcudnn8-dev

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
    pydub \
    git+https://github.com/m-bain/whisperx.git \
    git+https://github.com/openai/whisper.git # removed for whisperx for better timestamps


# ---- CHECK HUGGING FACE AUTH ----
echo "[*] Checking Hugging Face authentication..."
echo "Do you need to accept T/C to access a gated HF model?"
# https://huggingface.co/pyannote/segmentation-3.0
# https://huggingface.co/pyannote/speaker-diarization-3.1

HF_TOKEN="" # paste_your_token_here

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
    # Read the token from file, strip whitespace/newlines
    HF_TOKEN=$(<../.huggingface/token)
    echo "[✓] Hugging Face authentication found."
fi

# ---- CHECK HUGGING FACE MODEL ACCESS ----
echo "[*] Verifying access to pyannote/speaker-diarization-3.1..."

# if ! huggingface-cli whoami &>/dev/null; then
#     echo "[!] Hugging Face CLI not authenticated."
#     echo "    Run: huggingface-cli login"
#     exit 1
# fi

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

"$VENV_DIR/bin/python" preload-models.py "$HF_TOKEN"
