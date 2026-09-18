#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?missing execution mode}
shift

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="$ROOT_DIR/.venv"
DRY_RUN=0

if [[ ${1:-} == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

run() {
    if ((DRY_RUN)); then
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
        return
    fi
    "$@"
}

install_system_dependencies() {
    if command -v ffmpeg >/dev/null 2>&1 \
        && ldconfig -p 2>/dev/null | grep -q 'libsndfile\.so\.1'; then
        return
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Install ffmpeg and libsndfile1 with your system package manager." >&2
        exit 1
    fi

    local -a prefix=()
    if ((EUID != 0)); then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "Root privileges are required to install ffmpeg and libsndfile1." >&2
            exit 1
        fi
        prefix=(sudo)
    fi
    run "${prefix[@]}" apt-get update
    run "${prefix[@]}" apt-get install -y ffmpeg libsndfile1 python3-venv
}

if [[ $MODE != "cpu" && $MODE != "gpu" ]]; then
    echo "Unsupported execution mode: $MODE" >&2
    exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required." >&2
    exit 1
fi

if ((DRY_RUN)); then
    echo "Mode: $MODE"
    echo "Verify ffmpeg and libsndfile1; install them with apt-get when absent."
    if [[ $MODE == "gpu" ]]; then
        echo "Verify NVIDIA GPU and driver with nvidia-smi."
    fi
    run python3 -m venv "$VENV_DIR"
    run "$VENV_DIR/bin/python" -m pip install --upgrade pip
fi

if ((DRY_RUN == 0)); then
    install_system_dependencies
fi

if [[ ! -x $VENV_DIR/bin/python ]]; then
    run python3 -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
PIP=("$PYTHON" -m pip)

if [[ $MODE == "cpu" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    DEVICE="cpu"
else
    if ((DRY_RUN == 0)) && ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "No NVIDIA GPU driver found. Use run_cpu.sh or install the NVIDIA driver first." >&2
        exit 1
    fi
    if (( ! DRY_RUN )); then
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    fi
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    DEVICE="cuda"
fi

run "${PIP[@]}" install torch torchaudio --index-url "$TORCH_INDEX"
run "${PIP[@]}" install -e "$ROOT_DIR"

if [[ $MODE == "gpu" && $DRY_RUN -eq 0 ]]; then
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"; print(torch.cuda.get_device_name(0))'
fi

run "$PYTHON" "$ROOT_DIR/api_server.py" --device "$DEVICE" "$@"
