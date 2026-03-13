import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", help="Host GPU index to expose to the container.")
    parser.add_argument("--image", default="block-attention:latest")
    parser.add_argument("--name", default="block-attention-dev")
    parser.add_argument("--workspace", default=None, help="Path to the repo on the host.")
    parser.add_argument("--hf-cache", default=None, help="Host Hugging Face cache directory.")
    parser.add_argument("--uv-cache", default=None, help="Host uv cache directory.")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check)


def main() -> None:
    args = parse_args()
    root_dir = Path(args.workspace or Path(__file__).resolve().parents[1]).resolve()
    hf_cache = Path(args.hf_cache or Path.home() / ".cache" / "huggingface").resolve()
    uv_cache = Path(args.uv_cache or Path.home() / ".cache" / "uv").resolve()
    local_uid = os.getuid()
    local_gid = os.getgid()

    hf_cache.mkdir(parents=True, exist_ok=True)
    uv_cache.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        run([
            "docker", "build",
            "-t", args.image,
            "-f", str(root_dir / "docker" / "Dockerfile"),
            str(root_dir),
        ])

    existing = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if args.name in existing:
        run(["docker", "rm", "-f", args.name])

    run([
        "docker", "run", "-d",
        "--gpus", f"device={args.gpu}",
        "--name", args.name,
        "-v", f"{root_dir}:/workspace",
        "-v", f"{hf_cache}:/cache/huggingface",
        "-v", f"{uv_cache}:/cache/uv",
        "-e", "ENABLE_SSH=0",
        "-e", "BLOCK_ATTENTION_CUDA_DEVICE=cuda:0",
        "-e", f"LOCAL_UID={local_uid}",
        "-e", f"LOCAL_GID={local_gid}",
        "-e", "HF_HOME=/cache/huggingface",
        "-e", "HUGGINGFACE_HUB_CACHE=/cache/huggingface",
        "-e", "UV_CACHE_DIR=/cache/uv",
        "-w", "/workspace",
        args.image,
        "bash", "-lc", "sleep infinity",
    ])

    print(f"Container started: {args.name}")
    print(f"Host GPU exposed to container: {args.gpu}")
    print("Open VS Code on the server via Remote-SSH, then run:")
    print("  Dev Containers: Attach to Running Container...")
    print(f"  Select: {args.name}")
    print("After VS Code attaches, run:")
    print("  python scripts/reproduce_table1_block_ft.py --model ldsjmdy/Tulu3-Block-FT")


if __name__ == "__main__":
    main()
