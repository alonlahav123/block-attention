#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_SOURCE="ldsjmdy/Tulu3-Block-FT"
OUTPUT_ROOT="${ROOT_DIR}/outputs/table1_block_ft"
PORT=8080
NUM_LOCAL_ATTENTION_BLOCKS=10000
ATTN_IMPLEMENTATION="auto"
MAX_NEW_TOKENS=256
VENV_DIR="${ROOT_DIR}/.venv"
DATA_ROOT="${ROOT_DIR}/datahub"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
CUDA_DEVICE="${BLOCK_ATTENTION_CUDA_DEVICE:-cuda:0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_SOURCE="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-local-attention-blocks)
            NUM_LOCAL_ATTENTION_BLOCKS="$2"
            shift 2
            ;;
        --attn-implementation)
            ATTN_IMPLEMENTATION="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
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

PYTHON_BIN="${VENV_DIR}/bin/python"
MODEL_CACHE_DIR="${ROOT_DIR}/models"
SERVER_LOG="${OUTPUT_ROOT}/server.log"
SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

mkdir -p "${OUTPUT_ROOT}" "${MODEL_CACHE_DIR}"

INSTALL_FLASH_ATTN_VALUE="${INSTALL_FLASH_ATTN:-1}"
if [[ "${ATTN_IMPLEMENTATION}" != "auto" && "${ATTN_IMPLEMENTATION}" != "flash_attention_2" ]]; then
    INSTALL_FLASH_ATTN_VALUE=0
fi

export INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN_VALUE}"
bash "${ROOT_DIR}/scripts/prepare_table1_rag_eval.sh" \
    --data-root "${DATA_ROOT}" \
    --venv "${VENV_DIR}" \
    --cuda-device "${CUDA_DEVICE}"

if [[ -d "${MODEL_SOURCE}" ]]; then
    MODEL_DIR=$(cd "${MODEL_SOURCE}" && pwd)
else
    MODEL_DIR=$(
        "${PYTHON_BIN}" -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='${MODEL_SOURCE}', local_dir='${MODEL_CACHE_DIR}/$(basename "${MODEL_SOURCE}")'))"
    )
fi

check_torch_cuda() {
    "${PYTHON_BIN}" - <<'PY'
import os
import torch

print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"BLOCK_ATTENTION_CUDA_DEVICE={os.environ.get('BLOCK_ATTENTION_CUDA_DEVICE')}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch cannot access CUDA inside this container. "
        "Check Docker GPU runtime setup or the host driver/runtime compatibility."
    )
PY
}

wait_for_server() {
    echo "Waiting for server on port ${PORT}. Log: ${SERVER_LOG}"
    for attempt in $(seq 1 180); do
        if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            echo "Server process exited before becoming healthy. See ${SERVER_LOG}" >&2
            return 1
        fi
        if "${PYTHON_BIN}" -c "import requests; requests.post('http://127.0.0.1:${PORT}/generate', json={'blocks': ['<|user|>\\nYou are an intelligent AI assistant. Please answer questions based on the user\\'s instructions. Below are some reference documents that may help you in answering the user\\'s question.\\n\\n', '- Title: Warmup\\nWarmup\\n', '\\n\\nPlease write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant).\\nQuestion: What is the title?\\n<|assistant|>\\n']}, timeout=10).raise_for_status()" >/dev/null 2>&1; then
            echo "Server is healthy."
            return 0
        fi
        if (( attempt % 6 == 0 )); then
            echo "Still waiting for server startup... (${attempt}/180)"
        fi
        sleep 10
    done
    echo "Server did not become healthy within 30 minutes. See ${SERVER_LOG}" >&2
    return 1
}

make_smoke_file() {
    local input_fp="$1"
    local output_fp="$2"
    "${PYTHON_BIN}" -c "import json; from pathlib import Path; lines = Path('${input_fp}').read_text(encoding='utf-8').splitlines()[:5]; Path('${output_fp}').parent.mkdir(parents=True, exist_ok=True); Path('${output_fp}').write_text(''.join(line + '\n' for line in lines), encoding='utf-8')"
}

run_eval_summary() {
    local prefix="$1"
    shift
    "${PYTHON_BIN}" - "$prefix" "$@" <<'PY'
import json
import sys
from pathlib import Path

from rag_eval import evaluate_path

output_prefix = Path(sys.argv[1])
dataset_names = ["2wiki", "hqa", "nq", "tqa"]
paths = sys.argv[2:]

scores = {}
for dataset_name, path in zip(dataset_names, paths):
    result = evaluate_path(path)
    scores[dataset_name] = {
        "count": int(result["count"]),
        "best_subspan_em": result["best_subspan_em"],
    }

macro_average = sum(item["best_subspan_em"] for item in scores.values()) / len(scores)
summary = {
    "datasets": scores,
    "macro_average": macro_average,
}

output_prefix.parent.mkdir(parents=True, exist_ok=True)
output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

markdown_lines = [
    "| Dataset | Count | best_subspan_em |",
    "| --- | ---: | ---: |",
]
for dataset_name in dataset_names:
    item = scores[dataset_name]
    markdown_lines.append(
        f"| {dataset_name} | {item['count']} | {item['best_subspan_em']:.6f} |"
    )
markdown_lines.append(f"| macro_average | - | {macro_average:.6f} |")
output_prefix.with_suffix(".md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
PY
}

check_torch_cuda

"${PYTHON_BIN}" server/block_generate_server.py \
    --model "${MODEL_DIR}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "Started server pid ${SERVER_PID}"
wait_for_server

mkdir -p "${OUTPUT_ROOT}/smoke/generated" "${OUTPUT_ROOT}/smoke/inputs" "${OUTPUT_ROOT}/generated"

declare -A DATASETS=(
    [2wiki]="${DATA_ROOT}/rag/2wiki_eval/dataset"
    [hqa]="${DATA_ROOT}/rag/hqa_eval/dataset"
    [nq]="${DATA_ROOT}/rag/nq_eval/dataset"
    [tqa]="${DATA_ROOT}/rag/tqa_eval/dataset"
)

for dataset_name in 2wiki hqa nq tqa; do
    make_smoke_file "${DATASETS[$dataset_name]}" "${OUTPUT_ROOT}/smoke/inputs/${dataset_name}.jsonl"
    "${PYTHON_BIN}" scripts/run_rag_block_inference.py \
        --input "${OUTPUT_ROOT}/smoke/inputs/${dataset_name}.jsonl" \
        --output "${OUTPUT_ROOT}/smoke/generated/${dataset_name}.jsonl" \
        --server-url "http://127.0.0.1:${PORT}/generate" \
        --num-local-attention-blocks "${NUM_LOCAL_ATTENTION_BLOCKS}"
done

run_eval_summary "${OUTPUT_ROOT}/smoke/results" \
    "${OUTPUT_ROOT}/smoke/generated/2wiki.jsonl" \
    "${OUTPUT_ROOT}/smoke/generated/hqa.jsonl" \
    "${OUTPUT_ROOT}/smoke/generated/nq.jsonl" \
    "${OUTPUT_ROOT}/smoke/generated/tqa.jsonl"

for dataset_name in 2wiki hqa nq tqa; do
    "${PYTHON_BIN}" scripts/run_rag_block_inference.py \
        --input "${DATASETS[$dataset_name]}" \
        --output "${OUTPUT_ROOT}/generated/${dataset_name}.jsonl" \
        --server-url "http://127.0.0.1:${PORT}/generate" \
        --num-local-attention-blocks "${NUM_LOCAL_ATTENTION_BLOCKS}"
done

run_eval_summary "${OUTPUT_ROOT}/results" \
    "${OUTPUT_ROOT}/generated/2wiki.jsonl" \
    "${OUTPUT_ROOT}/generated/hqa.jsonl" \
    "${OUTPUT_ROOT}/generated/nq.jsonl" \
    "${OUTPUT_ROOT}/generated/tqa.jsonl"

echo "Finished Table 1 Block-FT reproduction."
echo "Smoke summary: ${OUTPUT_ROOT}/smoke/results.md"
echo "Final summary: ${OUTPUT_ROOT}/results.md"
