#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

cd "${ROOT_DIR}"

ensure_command() {
    local command_name="$1"
    local apt_package="${2:-$1}"

    if command -v "${command_name}" >/dev/null 2>&1; then
        return 0
    fi

    if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
        apt-get update
        apt-get install -y --no-install-recommends "${apt_package}"
        rm -rf /var/lib/apt/lists/*
        return 0
    fi

    echo "Missing required command '${command_name}'. Install package '${apt_package}' and rerun." >&2
    exit 1
}

patch_fid_for_local_env() {
    local fid_dir="$1"
    local preprocess_fp="${fid_dir}/src/preprocess.py"
    local get_data_fp="${fid_dir}/get-data.sh"

    if [[ ! -f "${preprocess_fp}" || ! -f "${get_data_fp}" ]]; then
        return 0
    fi

    FID_PREPROCESS_FP="${preprocess_fp}" FID_GET_DATA_FP="${get_data_fp}" FID_PYTHON_BIN="${PYTHON_BIN}" python3 - <<'PY'
from pathlib import Path
import os
import re

preprocess_path = Path(os.environ["FID_PREPROCESS_FP"])
get_data_path = Path(os.environ["FID_GET_DATA_FP"])
python_bin = os.environ["FID_PYTHON_BIN"]

preprocess_text = preprocess_path.read_text(encoding="utf-8")
patched_preprocess_text = preprocess_text.replace("import parser\n", "")
if patched_preprocess_text != preprocess_text:
    preprocess_path.write_text(patched_preprocess_text, encoding="utf-8")

get_data_text = get_data_path.read_text(encoding="utf-8")
patched_get_data_text = re.sub(
    r'(^\s*)python\s+src/preprocess\.py\s+\$DOWNLOAD\s+\$ROOT\s*$',
    rf'\1"{python_bin}" src/preprocess.py "$DOWNLOAD" "$ROOT"',
    get_data_text,
    flags=re.MULTILINE,
)
if patched_get_data_text != get_data_text:
    get_data_path.write_text(patched_get_data_text, encoding="utf-8")
PY
}

verify_python_environment() {
    local python_bin="$1"

    "${python_bin}" - <<'PY'
import importlib
import sys

required_modules = [
    "numpy",
    "pandas",
    "pyarrow",
    "requests",
    "torch",
    "transformers",
]

missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required Python modules: {', '.join(missing)}")
PY
}

DATA_ROOT="${ROOT_DIR}/datahub"
VENV_DIR="${ROOT_DIR}/.venv"
INSTALL_DEPS=1
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_PACKAGE="${TORCH_PACKAGE:-torch==2.6.0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
CUDA_DEVICE="${BLOCK_ATTENTION_CUDA_DEVICE:-cuda:0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --skip-install)
            INSTALL_DEPS=0
            shift
            ;;
        --cuda-visible-devices)
            CUDA_VISIBLE_DEVICES_VALUE="$2"
            shift 2
            ;;
        --cuda-device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi
export BLOCK_ATTENTION_CUDA_DEVICE="${CUDA_DEVICE}"

ensure_command wget

PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ "${INSTALL_DEPS}" -eq 0 && ! -x "${PYTHON_BIN}" ]]; then
    echo "Expected virtualenv python at ${PYTHON_BIN}. Remove --skip-install or create the venv first." >&2
    exit 1
fi

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
    uv venv --seed "${VENV_DIR}"
    uv pip install --python "${PYTHON_BIN}" --index-url "${TORCH_INDEX_URL}" "${TORCH_PACKAGE}"
    uv pip install --python "${PYTHON_BIN}" \
        accelerate \
        fire \
        flask \
        flask-cors \
        huggingface_hub \
        ninja \
        numpy \
        packaging \
        pandas \
        pyarrow \
        regex \
        requests \
        safetensors \
        sentencepiece \
        tiktoken \
        tqdm \
        "transformers>=4.56"

    if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
        uv pip install --python "${PYTHON_BIN}" \
            pip \
            setuptools \
            wheel
        uv pip install --python "${PYTHON_BIN}" --no-build-isolation flash-attn
    fi
fi

verify_python_environment "${PYTHON_BIN}"

mkdir -p "${DATA_ROOT}/2wiki" "${DATA_ROOT}/hqa" "${DATA_ROOT}/nq" "${DATA_ROOT}/rag" "${DATA_ROOT}/tqa"

if [[ ! -f "${DATA_ROOT}/2WikiMultihopQA/dev.parquet" ]]; then
    "${PYTHON_BIN}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='xanhho/2WikiMultihopQA', repo_type='dataset', local_dir='${DATA_ROOT}/2WikiMultihopQA')"
fi
ln -snf "${DATA_ROOT}/2WikiMultihopQA" "${DATA_ROOT}/2wiki"

if [[ ! -f "${DATA_ROOT}/hqa/hotpot_dev_distractor_v1.json" ]]; then
    curl -L "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" \
        -o "${DATA_ROOT}/hqa/hotpot_dev_distractor_v1.json"
fi

if [[ ! -d "${DATA_ROOT}/FiD" ]]; then
    git clone https://github.com/facebookresearch/FiD "${DATA_ROOT}/FiD"
fi
patch_fid_for_local_env "${DATA_ROOT}/FiD"

if [[ ! -f "${DATA_ROOT}/FiD/open_domain_data/NQ/test.json" || ! -f "${DATA_ROOT}/FiD/open_domain_data/TQA/test.json" ]]; then
    (
        cd "${DATA_ROOT}/FiD"
        bash get-data.sh
    )
fi

ln -snf "${DATA_ROOT}/FiD/open_domain_data/NQ/test.json" "${DATA_ROOT}/nq/test.json"
ln -snf "${DATA_ROOT}/FiD/open_domain_data/TQA/test.json" "${DATA_ROOT}/tqa/test.json"

"${PYTHON_BIN}" data_process/rag/hqa.py \
    --eval_fp "${DATA_ROOT}/hqa/hotpot_dev_distractor_v1.json" \
    --output_dir "${DATA_ROOT}/rag"

"${PYTHON_BIN}" data_process/rag/nq.py \
    --eval_fp "${DATA_ROOT}/nq/test.json" \
    --output_dir "${DATA_ROOT}/rag"

"${PYTHON_BIN}" data_process/rag/tqa.py \
    --eval_fp "${DATA_ROOT}/tqa/test.json" \
    --output_dir "${DATA_ROOT}/rag"

"${PYTHON_BIN}" data_process/rag/2wiki.py \
    --eval_fp "${DATA_ROOT}/2wiki/dev.parquet" \
    --output_dir "${DATA_ROOT}/rag"

echo "Prepared Table 1 RAG eval datasets under ${DATA_ROOT}/rag"
