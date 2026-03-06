import os


def get_cuda_device(default: str = "cuda:0") -> str:
    return os.environ.get("BLOCK_ATTENTION_CUDA_DEVICE", default)
