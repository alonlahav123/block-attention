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

ensure_python_alias() {
    local shim_dir="${ROOT_DIR}/.local/bin"

    if command -v python >/dev/null 2>&1; then
        return 0
    fi

    ensure_command python3 python3
    mkdir -p "${shim_dir}"
    ln -snf "$(command -v python3)" "${shim_dir}/python"
    export PATH="${shim_dir}:${PATH}"
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
ensure_python_alias

PYTHON_BIN="${VENV_DIR}/bin/python"

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
