#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

IMAGE_NAME="block-attention:latest"
CONTAINER_NAME="block-attention-dev"
HOST_GPU="0"
SSH_PORT="2222"
API_PORT="8080"
PASSWORD="${CONTAINER_PASSWORD:-blockattention}"
HF_CACHE_DIR="${HOME}/.cache/huggingface"
UV_CACHE_DIR="${HOME}/.cache/uv"
REBUILD_IMAGE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            HOST_GPU="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --ssh-port)
            SSH_PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --password)
            PASSWORD="$2"
            shift 2
            ;;
        --hf-cache)
            HF_CACHE_DIR="$2"
            shift 2
            ;;
        --uv-cache)
            UV_CACHE_DIR="$2"
            shift 2
            ;;
        --skip-build)
            REBUILD_IMAGE=0
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${HF_CACHE_DIR}" "${UV_CACHE_DIR}"

if [[ "${REBUILD_IMAGE}" -eq 1 ]]; then
    docker build -t "${IMAGE_NAME}" -f "${ROOT_DIR}/docker/Dockerfile" "${ROOT_DIR}"
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d \
    --gpus "device=${HOST_GPU}" \
    --name "${CONTAINER_NAME}" \
    -p "${SSH_PORT}:22" \
    -p "${API_PORT}:8080" \
    -v "${ROOT_DIR}:/workspace" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${UV_CACHE_DIR}:/root/.cache/uv" \
    -e ENABLE_SSH=1 \
    -e CONTAINER_PASSWORD="${PASSWORD}" \
    -e BLOCK_ATTENTION_CUDA_DEVICE="cuda:0" \
    -w /workspace \
    "${IMAGE_NAME}" \
    bash -lc "sleep infinity"

echo "Container started: ${CONTAINER_NAME}"
echo "Host GPU: ${HOST_GPU}"
echo "SSH: ssh root@localhost -p ${SSH_PORT}"
echo "Password: ${PASSWORD}"
echo "Docker exec: docker exec -it ${CONTAINER_NAME} bash"
echo "Inside the container, run:"
echo "  bash scripts/reproduce_table1_block_ft.sh --model ldsjmdy/Tulu3-Block-FT"
