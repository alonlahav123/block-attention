#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

cd "${ROOT_DIR}"

MODEL_SOURCE="ldsjmdy/Tulu3-Block-FT"
OUTPUT_ROOT="${ROOT_DIR}/outputs/table1_block_ft"
PORT=8080
NUM_LOCAL_ATTENTION_BLOCKS=10000
VENV_DIR="${ROOT_DIR}/.venv"
DATA_ROOT="${ROOT_DIR}/datahub"

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
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

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

bash "${ROOT_DIR}/scripts/prepare_table1_rag_eval.sh" --data-root "${DATA_ROOT}" --venv "${VENV_DIR}"

if [[ -d "${MODEL_SOURCE}" ]]; then
    MODEL_DIR=$(cd "${MODEL_SOURCE}" && pwd)
else
    MODEL_DIR=$(
        "${PYTHON_BIN}" -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='${MODEL_SOURCE}', local_dir='${MODEL_CACHE_DIR}/$(basename "${MODEL_SOURCE}")'))"
    )
fi

wait_for_server() {
    for _ in $(seq 1 180); do
        if "${PYTHON_BIN}" -c "import requests; requests.post('http://127.0.0.1:${PORT}/generate', json={'blocks': ['<|user|>\\nYou are an intelligent AI assistant. Please answer questions based on the user\\'s instructions. Below are some reference documents that may help you in answering the user\\'s question.\\n\\n', '- Title: Warmup\\nWarmup\\n', '\\n\\nPlease write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant).\\nQuestion: What is the title?\\n<|assistant|>\\n']}, timeout=10).raise_for_status()" >/dev/null 2>&1; then
            return 0
        fi
        sleep 10
    done
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

"${PYTHON_BIN}" server/block_generate_server.py --model "${MODEL_DIR}" --port "${PORT}" --dtype bfloat16 \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

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
