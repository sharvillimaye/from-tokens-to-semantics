#!/bin/bash
# Pod Setup Script for Mean Subspace Ablation Experiments
# Repository: https://github.com/sharvillimaye/team-aasa
#
# Usage:
#   # Option 1: Pass HF token as argument
#   ./pod_setup.sh YOUR_HF_TOKEN
#
#   # Option 2: Set HF_TOKEN environment variable before running
#   export HF_TOKEN="your_token_here"
#   ./pod_setup.sh
#
#   # Option 3: Interactive (will prompt for token)
#   ./pod_setup.sh
#
# The script will automatically:
#   - Install tmux if not present
#   - Create/attach to a tmux session named "ablation"
#   - Run the experiment inside tmux (safe from disconnects)

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
TMUX_SESSION_NAME="ablation"
WORKSPACE_DIR="/workspace"
REPO_NAME="experiment"
REPO_URL="https://github.com/sharvillimaye/team-aasa.git"
REPO_BRANCH="ablation"
EXPERIMENT_SCRIPT="ablation/mean_subspace_ablation.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${BLUE}==>${NC} ${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# ============================================================================
# Tmux Management
# ============================================================================
install_tmux() {
    if ! command -v tmux &> /dev/null; then
        print_step "Installing tmux..."
        apt-get update -qq
        apt-get install -y -qq tmux
        print_success "tmux installed"
    else
        print_success "tmux is already installed: $(tmux -V)"
    fi
}

# Check if we're inside tmux
is_inside_tmux() {
    [ -n "$TMUX" ]
}

# Check if our session exists
session_exists() {
    tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null
}

# Launch script inside tmux if not already there
ensure_tmux_session() {
    if is_inside_tmux; then
        print_success "Running inside tmux session: $(tmux display-message -p '#S')"
        return 0
    fi
    
    print_step "Tmux session management"
    
    if session_exists; then
        echo "Session '$TMUX_SESSION_NAME' already exists."
        echo ""
        echo "Options:"
        echo "  1) Attach to existing session (view progress)"
        echo "  2) Kill existing session and start fresh"
        echo "  3) Exit"
        echo ""
        read -p "Choose [1/2/3]: " -n 1 -r choice
        echo ""
        
        case $choice in
            1)
                echo "Attaching to existing session..."
                exec tmux attach-session -t "$TMUX_SESSION_NAME"
                ;;
            2)
                echo "Killing existing session..."
                tmux kill-session -t "$TMUX_SESSION_NAME"
                ;;
            3)
                echo "Exiting."
                exit 0
                ;;
            *)
                print_error "Invalid choice"
                exit 1
                ;;
        esac
    fi
    
    # Create new tmux session and run this script inside it
    echo "Creating new tmux session '$TMUX_SESSION_NAME'..."
    echo ""
    echo "The experiment will run inside tmux. You can safely disconnect with:"
    echo "  Ctrl+B, then D"
    echo ""
    echo "To reattach later:"
    echo "  tmux attach -t $TMUX_SESSION_NAME"
    echo ""
    sleep 2
    
    # Re-run this script inside tmux, passing along the HF_TOKEN
    if [ -n "$1" ]; then
        exec tmux new-session -s "$TMUX_SESSION_NAME" "HF_TOKEN='$1' $0 '$1'; exec bash"
    elif [ -n "$HF_TOKEN" ]; then
        exec tmux new-session -s "$TMUX_SESSION_NAME" "HF_TOKEN='$HF_TOKEN' $0; exec bash"
    else
        exec tmux new-session -s "$TMUX_SESSION_NAME" "$0; exec bash"
    fi
}

# ============================================================================
# Main Script Entry Point
# ============================================================================

# First, ensure tmux is installed
install_tmux

# Then ensure we're running inside tmux
ensure_tmux_session "$1"

# If we get here, we're inside tmux - continue with setup
print_step "Running inside tmux session: $TMUX_SESSION_NAME"
echo "You can safely disconnect with: Ctrl+B, then D"
echo "Reattach with: tmux attach -t $TMUX_SESSION_NAME"

# Gated models that require HuggingFace authentication
GATED_MODELS=(
    "meta-llama/Meta-Llama-3-8B"
    "google/gemma-2-2b"
    "google/gemma-2-9b"
    "mistralai/Mistral-7B-v0.3"
)

# All models to verify (including non-gated)
ALL_MODELS=(
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-1b-deduped"
    "EleutherAI/pythia-2.8b-deduped"
    "EleutherAI/pythia-6.9b-deduped"
    "allenai/OLMo-1B-hf"
    "allenai/OLMo-7B-hf"
    "google/gemma-2-2b"
    "meta-llama/Meta-Llama-3-8B"
    "google/gemma-2-9b"
    "mistralai/Mistral-7B-v0.3"
)

# ============================================================================
# Step 1: Navigate to workspace
# ============================================================================
print_step "Setting up workspace directory"
cd "$WORKSPACE_DIR" || { print_error "Could not cd to $WORKSPACE_DIR"; exit 1; }
print_success "Working directory: $(pwd)"

# ============================================================================
# Step 2: Clone repository (or pull if exists)
# ============================================================================
print_step "Cloning repository (branch: $REPO_BRANCH)"
if [ -d "$REPO_NAME" ]; then
    print_warning "Directory '$REPO_NAME' already exists"
    read -p "Delete and re-clone? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$REPO_NAME"
        git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_NAME"
        print_success "Repository cloned fresh (branch: $REPO_BRANCH)"
    else
        cd "$REPO_NAME"
        git fetch origin
        git checkout "$REPO_BRANCH" 2>/dev/null || git checkout -b "$REPO_BRANCH" "origin/$REPO_BRANCH"
        git pull origin "$REPO_BRANCH" || print_warning "Could not pull latest changes"
        print_success "Using existing repository (branch: $REPO_BRANCH)"
    fi
else
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_NAME"
    print_success "Repository cloned (branch: $REPO_BRANCH)"
fi

cd "$WORKSPACE_DIR/$REPO_NAME"

# ============================================================================
# Step 3: Install uv package manager
# ============================================================================
print_step "Installing uv package manager"
if command -v uv &> /dev/null; then
    print_success "uv is already installed: $(uv --version)"
else
    pip install uv
    print_success "uv installed"
fi

# ============================================================================
# Step 4: Create virtual environment and install dependencies
# ============================================================================
print_step "Setting up Python virtual environment"
if [ -d ".venv" ]; then
    print_warning ".venv already exists, using existing environment"
else
    uv venv
    print_success "Virtual environment created"
fi

# Activate the virtual environment
source .venv/bin/activate
print_success "Virtual environment activated: $(which python)"

print_step "Installing dependencies with uv sync"
uv sync
print_success "Dependencies installed"

# ============================================================================
# Step 5: HuggingFace Authentication
# ============================================================================
print_step "Setting up HuggingFace authentication"

# Get token from argument, environment variable, or prompt
HF_TOKEN="${1:-$HF_TOKEN}"

if [ -z "$HF_TOKEN" ]; then
    echo "HuggingFace token not provided via argument or HF_TOKEN env var."
    echo ""
    echo "You need a token with access to gated models:"
    for model in "${GATED_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    read -sp "Enter your HuggingFace token (input hidden): " HF_TOKEN
    echo ""
fi

if [ -z "$HF_TOKEN" ]; then
    print_error "No HuggingFace token provided. Cannot authenticate."
    print_warning "Continuing without authentication - gated models will fail!"
else
    # Use huggingface-cli for authentication (non-interactive)
    export HF_TOKEN="$HF_TOKEN"
    
    # Write token to cache for persistence
    mkdir -p ~/.cache/huggingface
    echo "$HF_TOKEN" > ~/.cache/huggingface/token
    
    # Also set the environment variable for the session
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HF_HOME="$WORKSPACE_DIR/.cache/huggingface"
    
    # Try to verify authentication
    if python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami()['name'])" 2>/dev/null; then
        print_success "HuggingFace authentication successful"
    else
        # Try alternative login method
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
        print_success "HuggingFace token configured"
    fi
fi

# ============================================================================
# Step 6: Verify model access
# ============================================================================
print_step "Verifying access to models (downloading configs only)"

# Create a Python script to verify model access
cat > /tmp/verify_models.py << 'VERIFY_SCRIPT'
import sys
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
import os

models = sys.argv[1:]
api = HfApi()

print("\nVerifying model access:")
print("=" * 60)

all_ok = True
results = []

for model in models:
    try:
        # Try to download just the config file to verify access
        config_path = hf_hub_download(
            repo_id=model,
            filename="config.json",
            token=os.environ.get("HF_TOKEN"),
        )
        results.append((model, "✓", "Access OK"))
    except GatedRepoError:
        results.append((model, "✗", "GATED - Need to accept license at huggingface.co"))
        all_ok = False
    except RepositoryNotFoundError:
        results.append((model, "✗", "NOT FOUND"))
        all_ok = False
    except Exception as e:
        results.append((model, "?", f"Error: {str(e)[:50]}"))
        all_ok = False

# Print results
max_model_len = max(len(r[0]) for r in results)
for model, status, message in results:
    print(f"  {status} {model:<{max_model_len}}  {message}")

print("=" * 60)

if not all_ok:
    print("\n⚠️  Some models are not accessible!")
    print("   For gated models, visit the model page on HuggingFace and accept the license.")
    print("   Example: https://huggingface.co/meta-llama/Meta-Llama-3-8B")
    sys.exit(1)
else:
    print("\n✓ All models are accessible!")
    sys.exit(0)
VERIFY_SCRIPT

python /tmp/verify_models.py "${ALL_MODELS[@]}" || {
    print_warning "Some models are not accessible. See above for details."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Aborting. Please fix model access issues first."
        exit 1
    fi
}

# ============================================================================
# Step 7: System information
# ============================================================================
print_step "System Information"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "  GPU Memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")')"
fi

# ============================================================================
# Step 8: Ready to run
# ============================================================================
print_step "Setup complete!"
echo ""
echo "To run the experiment:"
echo ""
echo "  cd $WORKSPACE_DIR/$REPO_NAME"
echo "  source .venv/bin/activate"
echo "  uv run python $EXPERIMENT_SCRIPT"
echo ""
echo "Or run directly with:"
echo ""
echo "  cd $WORKSPACE_DIR/$REPO_NAME && source .venv/bin/activate && uv run python $EXPERIMENT_SCRIPT"
echo ""

# Ask if user wants to run now
read -p "Run the experiment now? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    print_step "Starting experiment..."
    echo ""
    
    # Set environment variables for better CUDA error handling
    export CUDA_LAUNCH_BLOCKING=0
    export HF_HOME="$WORKSPACE_DIR/.cache/huggingface"
    
    # Run the experiment
    uv run python "$EXPERIMENT_SCRIPT"
fi

