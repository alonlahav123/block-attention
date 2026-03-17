import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ldsjmdy/Tulu3-RAG")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--parallelism", default="data", choices=["data", "tensor"])
    parser.add_argument("--request-concurrency", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--venv", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--server-backend", default="auto", choices=["auto", "repo", "upstream"])
    parser.add_argument("--benchmark-parallelism", action="store_true")
    parser.add_argument("--benchmark-examples", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]

    cmd = [
        "bash",
        str(root_dir / "scripts" / "reproduce_table1_rag_baseline.sh"),
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--gpu-ids",
        args.gpu_ids,
        "--parallelism",
        args.parallelism,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--server-backend",
        args.server_backend,
        "--benchmark-examples",
        str(args.benchmark_examples),
    ]

    if args.output_root is not None:
        cmd.extend(["--output-root", args.output_root])
    if args.request_concurrency is not None:
        cmd.extend(["--request-concurrency", str(args.request_concurrency)])
    if args.venv is not None:
        cmd.extend(["--venv", args.venv])
    if args.data_root is not None:
        cmd.extend(["--data-root", args.data_root])
    if args.benchmark_parallelism:
        cmd.append("--benchmark-parallelism")
    if args.resume:
        cmd.append("--resume")

    subprocess.run(cmd, cwd=root_dir, check=True)


if __name__ == "__main__":
    main()
