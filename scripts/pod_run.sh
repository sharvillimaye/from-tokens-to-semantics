#!/bin/bash
# Non-interactive Pod Setup and Run Script
# 
# Usage:
#   HF_TOKEN="your_token" ./pod_run.sh
#
# Or copy-paste this one-liner:
#   curl -sL https://raw.githubusercontent.com/sharvillimaye/team-aasa/ablation/scripts/pod_run.sh | HF_TOKEN="your_token" bash
#
# The script automatically runs inside tmux for safety against disconnects.

set -e

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_NAME="experiment"
REPO_URL="https://github.com/sharvillimaye/team-aasa.git"
REPO_BRANCH="ablation"
TMUX_SESSION_NAME="ablation"

# ============================================================================
# Tmux Management (non-interactive)
# ============================================================================
install_tmux() {
    if ! command -v tmux &> /dev/null; then
        echo "Installing tmux..."
        apt-get update -qq
        apt-get install -y -qq tmux
    fi
}

# Install tmux first
install_tmux

# If not inside tmux, re-launch inside tmux
if [ -z "$TMUX" ]; then
    # Kill existing session if any
    tmux kill-session -t "$TMUX_SESSION_NAME" 2>/dev/null || true
    
    echo "=========================================="
    echo "Launching in tmux session: $TMUX_SESSION_NAME"
    echo "=========================================="
    echo ""
    echo "To detach (keep running): Ctrl+B, then D"
    echo "To reattach: tmux attach -t $TMUX_SESSION_NAME"
    echo ""
    sleep 2
    
    # Re-run this script inside tmux
    exec tmux new-session -s "$TMUX_SESSION_NAME" "HF_TOKEN='$HF_TOKEN' $0; exec bash"
fi

echo "=========================================="
echo "Pod Setup for Mean Subspace Ablation"
echo "Running in tmux: $TMUX_SESSION_NAME"
echo "=========================================="

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is required"
    echo "Usage: HF_TOKEN='your_token' ./pod_run.sh"
    exit 1
fi

# Navigate to workspace
cd "$WORKSPACE_DIR"
echo "✓ Working in: $WORKSPACE_DIR"

# Clone or update repository
if [ -d "$REPO_NAME" ]; then
    echo "Repository exists, pulling latest from $REPO_BRANCH branch..."
    cd "$REPO_NAME"
    git fetch origin
    git checkout "$REPO_BRANCH" 2>/dev/null || git checkout -b "$REPO_BRANCH" "origin/$REPO_BRANCH"
    git pull origin "$REPO_BRANCH" || true
else
    echo "Cloning repository (branch: $REPO_BRANCH)..."
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_NAME"
    cd "$REPO_NAME"
fi
echo "✓ Repository ready (branch: $REPO_BRANCH)"

# Install uv if needed
if ! command -v uv &> /dev/null; then
    pip install uv
fi
echo "✓ uv installed"

# Setup virtual environment
[ ! -d ".venv" ] && uv venv
source .venv/bin/activate
uv sync
echo "✓ Dependencies installed"

# Configure HuggingFace authentication
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME="$WORKSPACE_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"
echo "$HF_TOKEN" > "$HF_HOME/token"
echo "✓ HuggingFace configured"

# Verify GPU
python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)') if torch.cuda.is_available() else print('⚠ No GPU detected')"

# Run experiment
echo ""
echo "=========================================="
echo "Starting Experiment"
echo "=========================================="
uv run python ablation/mean_subspace_ablation.py

