#!/usr/bin/env bash
set -euo pipefail

# ─── VidCore Fresh VM Setup ──────────────────────────────────────────────────
# Usage:
#   ./scripts/setup.sh                    # install deps + start vLLM
#   ./scripts/setup.sh --serve-only       # just start vLLM (assumes model cached)
#   ./scripts/setup.sh --no-serve         # install deps only, don't start server
#   ./scripts/setup.sh --model <id>       # use custom HuggingFace model
#   VLLM_PORT=8080 ./scripts/setup.sh     # custom port (default 8000)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVE=true
INSTALL=true

for arg in "$@"; do
    case "$arg" in
        --serve-only) INSTALL=false ;;
        --no-serve)   SERVE=false ;;
        --model)      shift; MODEL="$1" ;;
        --port)       shift; VLLM_PORT="$1" ;;
    esac
done

echo "══════════════════════════════════════════"
echo " VidCore Setup"
echo "══════════════════════════════════════════"
echo "  Model:    $MODEL"
echo "  vLLM port: $VLLM_PORT"
echo "  Project:  $PROJECT_ROOT"
echo "══════════════════════════════════════════"
echo ""

# ─── OS detection ────────────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="macos" ;;
        *)       echo "Unsupported OS: $(uname -s)"; exit 1 ;;
    esac
    echo "  Detected OS: $OS"
}

# ─── Python check ────────────────────────────────────────────────────────────
check_python() {
    if command -v python3 &>/dev/null; then
        PY="python3"
    elif command -v python &>/dev/null; then
        PY="python"
    else
        echo "ERROR: Python 3 not found. Install it first."
        echo "  Ubuntu: sudo apt install -y python3 python3-pip python3-venv"
        echo "  macOS:  brew install python@3.12"
        exit 1
    fi

    PY_VER=$("$PY" --version 2>&1 | grep -oP '\d+\.\d+')
    echo "  Python: $PY_VER ($PY)"
}

# ─── GPU check ───────────────────────────────────────────────────────────────
check_gpu() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "  GPU: $GPU_COUNT found"
        HAS_GPU=true
    else
        echo "  GPU: none (CPU-only mode)"
        HAS_GPU=false
    fi
}

# ─── Install system deps ──────────────────────────────────────────────────────
install_system_deps() {
    echo ""
    echo "── System dependencies ──"
    if [ "$OS" = "linux" ]; then
        # OpenCV + vLLM system deps for Ubuntu
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y -qq \
                libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
                libgomp1 ffmpeg 2>/dev/null || true
            echo "  Ubuntu deps installed"
        fi
    elif [ "$OS" = "macos" ]; then
        if command -v brew &>/dev/null; then
            brew install ffmpeg 2>/dev/null || true
            echo "  macOS deps installed"
        fi
    fi
}

# ─── Setup venv ──────────────────────────────────────────────────────────────
setup_venv() {
    echo ""
    echo "── Virtual environment ──"
    if [ ! -d "$VENV_DIR" ]; then
        "$PY" -m venv "$VENV_DIR"
        echo "  Created .venv"
    else
        echo "  .venv already exists"
    fi
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
}

# ─── Install Python deps ─────────────────────────────────────────────────────
install_python_deps() {
    echo ""
    echo "── Python dependencies ──"

    pip install -r "$PROJECT_ROOT/requirements.txt" -q
    echo "  Project deps installed"

    # vLLM with appropriate backend
    if [ "$HAS_GPU" = true ]; then
        pip install vllm -q
        echo "  vLLM (CUDA) installed"
    else
        pip install vllm -q
        echo "  vLLM (CPU) installed"
    fi
}

# ─── Download model ──────────────────────────────────────────────────────────
download_model() {
    echo ""
    echo "── Model download ──"
    echo "  Model: $MODEL"

    # vLLM auto-downloads on first serve; pre-fetch to avoid cold-start
    if python -c "from huggingface_hub import snapshot_download" 2>/dev/null; then
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', resume_download=True)
" && echo "  Model cached" || echo "  Model will be downloaded on first vLLM start"
    else
        echo "  huggingface_hub not installed; model will download on first vLLM start"
    fi
}

# ─── Start vLLM ──────────────────────────────────────────────────────────────
start_vllm() {
    echo ""
    echo "── Starting vLLM server ──"

    VLLM_CMD="vllm serve $MODEL"
    VLLM_CMD="$VLLM_CMD --host 0.0.0.0"
    VLLM_CMD="$VLLM_CMD --port $VLLM_PORT"
    VLLM_CMD="$VLLM_CMD --dtype ${DTYPE:-auto}"
    VLLM_CMD="$VLLM_CMD --max-model-len ${MAX_MODEL_LEN:-32768}"

    if [ "$HAS_GPU" = false ]; then
        VLLM_CMD="$VLLM_CMD --cpu-offload-gb 16"
        echo "  (CPU-only mode)"
    else
        VLLM_CMD="$VLLM_CMD --gpu-memory-utilization ${GPU_MEM_UTIL:-0.90}"
    fi

    echo ""
    echo "  Command: $VLLM_CMD"
    echo ""
    echo "  Server will be available at: http://localhost:$VLLM_PORT"
    echo "  API endpoint: http://localhost:$VLLM_PORT/v1/chat/completions"
    echo ""

    # Update config.yaml with local endpoint
    if command -v yq &>/dev/null; then
        yq -i ".vllm_endpoint = \"http://localhost:$VLLM_PORT/v1/chat/completions\"" \
            "$PROJECT_ROOT/config.yaml" 2>/dev/null || true
    else
        # fallback: python YAML update
        python -c "
import yaml
cfg = yaml.safe_load(open('$PROJECT_ROOT/config.yaml'))
cfg['vllm_endpoint'] = 'http://localhost:$VLLM_PORT/v1/chat/completions'
yaml.dump(cfg, open('$PROJECT_ROOT/config.yaml', 'w'), default_flow_style=False)
" && echo "  config.yaml updated with local endpoint"
    fi

    echo ""
    echo "══════════════════════════════════════════"
    echo " Setup complete!"
    echo ""
    echo " Run VidCore:"
    echo "   source .venv/bin/activate"
    echo "   python main.py analyze videos/1.mp4"
    echo ""
    echo " Run tests:"
    echo "   python tests.py"
    echo "══════════════════════════════════════════"
    echo ""

    echo "Starting vLLM (Ctrl+C to stop)..."
    exec $VLLM_CMD
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    detect_os
    check_python
    check_gpu

    if [ "$INSTALL" = true ]; then
        install_system_deps
        setup_venv
        install_python_deps
        download_model
    fi

    if [ "$SERVE" = true ]; then
        start_vllm
    else
        echo ""
        echo "══════════════════════════════════════════"
        echo " Deps installed. Start vLLM manually:"
        echo "   source .venv/bin/activate"
        echo "   vllm serve $MODEL --host 0.0.0.0 --port $VLLM_PORT --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.90"
        echo "══════════════════════════════════════════"
    fi
}

main
