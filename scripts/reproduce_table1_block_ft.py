import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-local-attention-blocks", type=int, default=10000)
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--venv", default=None)
    parser.add_argument("--data-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]

    cmd = [
        "bash",
        str(root_dir / "scripts" / "reproduce_table1_block_ft.sh"),
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--num-local-attention-blocks",
        str(args.num_local_attention_blocks),
        "--attn-implementation",
        args.attn_implementation,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]

    if args.output_root is not None:
        cmd.extend(["--output-root", args.output_root])
    if args.venv is not None:
        cmd.extend(["--venv", args.venv])
    if args.data_root is not None:
        cmd.extend(["--data-root", args.data_root])

    subprocess.run(cmd, cwd=root_dir, check=True)


if __name__ == "__main__":
    main()
