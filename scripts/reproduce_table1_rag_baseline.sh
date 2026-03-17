#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_SOURCE="ldsjmdy/Tulu3-RAG"
OUTPUT_ROOT="${ROOT_DIR}/outputs/table1_tulu3_rag"
PORT=8080
GPU_IDS="0,1"
PARALLELISM="data"
REQUEST_CONCURRENCY=""
MAX_NEW_TOKENS=128
VENV_DIR="${ROOT_DIR}/.venv-table1-vllm"
DATA_ROOT="${ROOT_DIR}/datahub"
SERVER_BACKEND="auto"
BENCHMARK_PARALLELISM=0
BENCHMARK_EXAMPLES=64
RESUME=0
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
CUDA_DEVICE="${BLOCK_ATTENTION_CUDA_DEVICE:-cuda:0}"
SERVER_PID=""
SERVER_LOG=""
SERVER_BACKEND_USED=""

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
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --parallelism)
            PARALLELISM="$2"
            shift 2
            ;;
        --request-concurrency)
            REQUEST_CONCURRENCY="$2"
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
        --server-backend)
            SERVER_BACKEND="$2"
            shift 2
            ;;
        --benchmark-parallelism)
            BENCHMARK_PARALLELISM=1
            shift
            ;;
        --benchmark-examples)
            BENCHMARK_EXAMPLES="$2"
            shift 2
            ;;
        --resume)
            RESUME=1
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

PYTHON_BIN="${VENV_DIR}/bin/python"
MODEL_CACHE_DIR="${ROOT_DIR}/models"
OUTPUT_ROOT=$(python3 - <<'PY' "$OUTPUT_ROOT"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)
mkdir -p "${MODEL_CACHE_DIR}"

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

parse_gpu_ids() {
    local value="$1"
    local -n out_ref=$2
    IFS=',' read -r -a out_ref <<< "$value"
    if [[ "${#out_ref[@]}" -eq 0 ]]; then
        echo "Expected at least one gpu id" >&2
        exit 1
    fi
}

gpu_count() {
    local gpu_array=()
    parse_gpu_ids "$GPU_IDS" gpu_array
    echo "${#gpu_array[@]}"
}

NUM_GPUS=$(gpu_count)
if [[ -z "${REQUEST_CONCURRENCY}" ]]; then
    REQUEST_CONCURRENCY="${NUM_GPUS}"
fi

if [[ "${PARALLELISM}" != "data" && "${PARALLELISM}" != "tensor" ]]; then
    echo "--parallelism must be one of: data, tensor" >&2
    exit 1
fi

if [[ "${SERVER_BACKEND}" != "auto" && "${SERVER_BACKEND}" != "repo" && "${SERVER_BACKEND}" != "upstream" ]]; then
    echo "--server-backend must be one of: auto, repo, upstream" >&2
    exit 1
fi

if [[ "${RESUME}" -eq 0 && -e "${OUTPUT_ROOT}" ]]; then
    echo "Refusing to write into existing output root without --resume: ${OUTPUT_ROOT}" >&2
    exit 1
fi

if [[ "${RESUME}" -eq 1 && ! -d "${OUTPUT_ROOT}" ]]; then
    echo "Cannot resume because output root does not exist: ${OUTPUT_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

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
        "PyTorch cannot access CUDA. Check the runtime before starting vLLM."
    )
PY
}

resolve_parallelism_args() {
    local mode="$1"
    local tp_size=1
    local dp_size=1
    if [[ "$mode" == "data" ]]; then
        dp_size="${NUM_GPUS}"
    else
        tp_size="${NUM_GPUS}"
    fi
    echo "$tp_size $dp_size"
}

wait_for_server() {
    local timeout_seconds="${1:-1800}"
    local attempts=$(( timeout_seconds / 5 ))
    echo "Waiting for ${SERVER_BACKEND_USED} server on port ${PORT}. Log: ${SERVER_LOG}"
    for attempt in $(seq 1 "${attempts}"); do
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
            return 1
        fi
        if "${PYTHON_BIN}" -c "import requests; requests.get('http://127.0.0.1:${PORT}/health', timeout=10).raise_for_status()" >/dev/null 2>&1; then
            echo "Server is healthy."
            return 0
        fi
        if (( attempt % 12 == 0 )); then
            echo "Still waiting for server startup... (${attempt}/$attempts)"
        fi
        sleep 5
    done
    return 1
}

start_server_with_backend() {
    local backend="$1"
    local mode="$2"
    read -r tp_size dp_size <<< "$(resolve_parallelism_args "$mode")"
    SERVER_BACKEND_USED="$backend"
    SERVER_LOG="${OUTPUT_ROOT}/server_${backend}_${mode}.log"

    local base_args=(
        --model "${MODEL_DIR}"
        --host 0.0.0.0
        --port "${PORT}"
        --dtype bfloat16
        --tokenizer-mode slow
        --tensor-parallel-size "${tp_size}"
        --max-model-len 4096
    )
    if [[ "${dp_size}" != "1" ]]; then
        base_args+=(--data-parallel-size "${dp_size}")
    fi

    : > "${SERVER_LOG}"
    if [[ "$backend" == "repo" ]]; then
        "${PYTHON_BIN}" server/vllm_server.py "${base_args[@]}" >"${SERVER_LOG}" 2>&1 &
    else
        "${PYTHON_BIN}" -m vllm.entrypoints.api_server "${base_args[@]}" >"${SERVER_LOG}" 2>&1 &
    fi
    SERVER_PID=$!
    echo "Started ${backend} server pid ${SERVER_PID} (${mode}; tp=${tp_size}, dp=${dp_size})"

    if wait_for_server; then
        return 0
    fi

    echo "${backend} server failed to become healthy. See ${SERVER_LOG}" >&2
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    SERVER_PID=""
    return 1
}

start_server() {
    local mode="$1"
    case "${SERVER_BACKEND}" in
        repo)
            start_server_with_backend repo "$mode"
            ;;
        upstream)
            start_server_with_backend upstream "$mode"
            ;;
        auto)
            if start_server_with_backend repo "$mode"; then
                return 0
            fi
            echo "Falling back to upstream vLLM api_server"
            start_server_with_backend upstream "$mode"
            ;;
    esac
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    SERVER_PID=""
}

make_smoke_file() {
    local input_fp="$1"
    local output_fp="$2"
    "${PYTHON_BIN}" -c "from pathlib import Path; lines = Path('${input_fp}').read_text(encoding='utf-8').splitlines()[:5]; Path('${output_fp}').parent.mkdir(parents=True, exist_ok=True); Path('${output_fp}').write_text(''.join(line + '\n' for line in lines), encoding='utf-8')"
}

run_prompt_inference() {
    local input_fp="$1"
    local output_fp="$2"
    local done_marker="$3"
    local metrics_fp="$4"
    shift 4

    "${PYTHON_BIN}" scripts/run_rag_prompt_inference.py \
        --input "${input_fp}" \
        --output "${output_fp}" \
        --done-marker "${done_marker}" \
        --metrics-output "${metrics_fp}" \
        --server-url "http://127.0.0.1:${PORT}/generate" \
        --request-concurrency "${REQUEST_CONCURRENCY}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        "$@"
}

write_benchmark_summary() {
    local benchmark_dir="$1"
    "${PYTHON_BIN}" - <<'PY' "$benchmark_dir"
from pathlib import Path
import json
import sys

benchmark_dir = Path(sys.argv[1])
rows = []
for mode in ["data", "tensor"]:
    metrics_path = benchmark_dir / mode / "metrics.json"
    if not metrics_path.exists():
        continue
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows.append((mode, payload))

if not rows:
    raise SystemExit(0)

best_mode = max(rows, key=lambda item: item[1]["examples_per_second"])[0]
summary_path = benchmark_dir / "summary.md"
lines = [
    "| Mode | Count | Wall Seconds | Examples/s | Median Latency (s) |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for mode, payload in rows:
    lines.append(
        f"| {mode} | {payload['count']} | {payload['wall_seconds']:.2f} | "
        f"{payload['examples_per_second']:.4f} | {payload['median_request_latency_seconds']:.4f} |"
    )
lines.append(f"\nRecommended mode from this benchmark: `{best_mode}`")
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_path)
PY
}

export INSTALL_FLASH_ATTN=0
export INSTALL_VLLM=1
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

check_torch_cuda

if [[ "${BENCHMARK_PARALLELISM}" -eq 1 && "${NUM_GPUS}" -gt 1 ]]; then
    BENCHMARK_DIR="${OUTPUT_ROOT}/benchmark"
    mkdir -p "${BENCHMARK_DIR}"
    for mode in data tensor; do
        stop_server
        start_server "$mode"
        mkdir -p "${BENCHMARK_DIR}/${mode}"
        run_prompt_inference \
            "${DATA_ROOT}/rag/2wiki_eval/dataset" \
            "${BENCHMARK_DIR}/${mode}/2wiki.jsonl" \
            "${BENCHMARK_DIR}/${mode}/2wiki.done" \
            "${BENCHMARK_DIR}/${mode}/metrics.json" \
            --start-index 0 \
            --end-index "${BENCHMARK_EXAMPLES}"
    done
    stop_server
    BENCHMARK_SUMMARY=$(write_benchmark_summary "${BENCHMARK_DIR}")
    echo "Benchmark summary: ${BENCHMARK_SUMMARY}"
fi

start_server "${PARALLELISM}"

mkdir -p "${OUTPUT_ROOT}/smoke/generated" "${OUTPUT_ROOT}/smoke/inputs" "${OUTPUT_ROOT}/generated"

declare -A DATASETS=(
    [2wiki]="${DATA_ROOT}/rag/2wiki_eval/dataset"
    [hqa]="${DATA_ROOT}/rag/hqa_eval/dataset"
    [nq]="${DATA_ROOT}/rag/nq_eval/dataset"
    [tqa]="${DATA_ROOT}/rag/tqa_eval/dataset"
)

for dataset_name in 2wiki hqa nq tqa; do
    make_smoke_file "${DATASETS[$dataset_name]}" "${OUTPUT_ROOT}/smoke/inputs/${dataset_name}.jsonl"
    run_prompt_inference \
        "${OUTPUT_ROOT}/smoke/inputs/${dataset_name}.jsonl" \
        "${OUTPUT_ROOT}/smoke/generated/${dataset_name}.jsonl" \
        "${OUTPUT_ROOT}/smoke/generated/${dataset_name}.done" \
        "${OUTPUT_ROOT}/smoke/generated/${dataset_name}.metrics.json"
done

"${PYTHON_BIN}" scripts/write_table1_summary.py \
    --output-prefix "${OUTPUT_ROOT}/smoke/results" \
    --2wiki "${OUTPUT_ROOT}/smoke/generated/2wiki.jsonl" \
    --hqa "${OUTPUT_ROOT}/smoke/generated/hqa.jsonl" \
    --nq "${OUTPUT_ROOT}/smoke/generated/nq.jsonl" \
    --tqa "${OUTPUT_ROOT}/smoke/generated/tqa.jsonl"

for dataset_name in 2wiki hqa nq tqa; do
    run_prompt_inference \
        "${DATASETS[$dataset_name]}" \
        "${OUTPUT_ROOT}/generated/${dataset_name}.jsonl" \
        "${OUTPUT_ROOT}/generated/${dataset_name}.done" \
        "${OUTPUT_ROOT}/generated/${dataset_name}.metrics.json"
done

stop_server

"${PYTHON_BIN}" scripts/write_table1_summary.py \
    --output-prefix "${OUTPUT_ROOT}/results" \
    --2wiki "${OUTPUT_ROOT}/generated/2wiki.jsonl" \
    --hqa "${OUTPUT_ROOT}/generated/hqa.jsonl" \
    --nq "${OUTPUT_ROOT}/generated/nq.jsonl" \
    --tqa "${OUTPUT_ROOT}/generated/tqa.jsonl"

echo "Finished Table 1 Tulu3-RAG reproduction."
echo "Final summary: ${OUTPUT_ROOT}/results.md"
