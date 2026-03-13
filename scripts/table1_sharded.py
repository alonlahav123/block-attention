import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_ORDER = ["2wiki", "hqa", "nq", "tqa"]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "table1_sharded"


@dataclass
class ActiveChunkProcess:
    worker_id: str
    chunk_id: str
    process: subprocess.Popen[Any]
    log_handle: Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("Expected at least one GPU id")
    return gpu_ids


def slugify(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_"}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    return slug or "table1-sharded"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest)


def is_process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_jsonl_with_recovery(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    recovered: list[dict[str, Any]] = []
    file_size = path.stat().st_size
    last_good_offset = 0

    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                recovered.append(json.loads(line.decode("utf-8")))
                last_good_offset = handle.tell()
            except Exception:
                if handle.tell() != file_size:
                    raise ValueError(f"Malformed JSON before EOF in {path}")
                with path.open("rb+") as writable_handle:
                    writable_handle.truncate(line_start)
                break

    return recovered


def count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def count_recovered_output(path: Path) -> int:
    return len(read_jsonl_with_recovery(path))


def ensure_under_workspace(path: Path) -> None:
    try:
        path.resolve().relative_to(ROOT_DIR)
    except ValueError as exc:
        raise ValueError(f"Path must live under the repo workspace: {path}") from exc


def to_container_workspace_path(path: Path) -> str:
    ensure_under_workspace(path)
    relative = path.resolve().relative_to(ROOT_DIR)
    return str(Path("/workspace") / relative)


def docker(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
    stdout: Any = None,
    stderr: Any = None,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=capture_output,
        text=text,
        stdout=stdout,
        stderr=stderr,
    )


def docker_build(image: str) -> None:
    docker(
        [
            "build",
            "-t",
            image,
            "-f",
            str(ROOT_DIR / "docker" / "Dockerfile"),
            str(ROOT_DIR),
        ]
    )


def docker_remove_container(name: str) -> None:
    docker(["rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=False)


def docker_container_running(name: str) -> bool:
    result = docker(
        ["inspect", "-f", "{{.State.Running}}", name],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def host_mount_env(
    hf_cache: Path,
    uv_cache: Path,
    extra_mounts: list[tuple[Path, str, bool]] | None = None,
) -> list[str]:
    hf_cache.mkdir(parents=True, exist_ok=True)
    uv_cache.mkdir(parents=True, exist_ok=True)
    mounts = [
        "-v", f"{ROOT_DIR}:/workspace",
        "-v", f"{hf_cache}:/cache/huggingface",
        "-v", f"{uv_cache}:/cache/uv",
    ]
    for host_path, container_path, read_only in extra_mounts or []:
        host_path = host_path.resolve()
        mode = ":ro" if read_only else ""
        mounts.extend(["-v", f"{host_path}:{container_path}{mode}"])
    return mounts


def container_env_args() -> list[str]:
    return [
        "-e", f"LOCAL_UID={os.getuid()}",
        "-e", f"LOCAL_GID={os.getgid()}",
        "-e", "ENABLE_SSH=0",
        "-e", "BLOCK_ATTENTION_CUDA_DEVICE=cuda:0",
        "-e", "HF_HOME=/cache/huggingface",
        "-e", "HUGGINGFACE_HUB_CACHE=/cache/huggingface",
        "-e", "UV_CACHE_DIR=/cache/uv",
        "-w", "/workspace",
    ]


def resolve_model_spec(model: str) -> dict[str, str]:
    candidate = Path(model)
    if candidate.is_dir():
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(ROOT_DIR)
            return {
                "mode": "workspace_path",
                "host_path": str(resolved),
                "container_path": str(Path("/workspace") / relative),
            }
        except ValueError:
            return {
                "mode": "external_path",
                "host_path": str(resolved),
                "container_path": "/mounted_model",
            }

    return {
        "mode": "hf_repo",
        "repo_id": model,
        "container_path": str(Path("/workspace/models") / Path(model).name),
    }


def worker_container_name(run_slug: str, gpu_id: str) -> str:
    return f"table1-sharded-{run_slug}-gpu{gpu_id}"


def prep_container_name(run_slug: str) -> str:
    return f"table1-sharded-{run_slug}-prep"


def default_output_root() -> Path:
    return DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")


def create_manifest(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root().resolve()
    ensure_under_workspace(output_root)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists at {manifest_path}. Use resume instead.")

    output_root.mkdir(parents=True, exist_ok=True)
    run_slug = slugify(output_root.name)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    hf_cache = Path(args.hf_cache or Path.home() / ".cache" / "huggingface").resolve()
    uv_cache = Path(args.uv_cache or Path.home() / ".cache" / "uv").resolve()
    model_spec = resolve_model_spec(args.model)

    workers = {}
    for gpu_id in gpu_ids:
        worker_id = f"gpu{gpu_id}"
        workers[worker_id] = {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "container_name": worker_container_name(run_slug=run_slug, gpu_id=gpu_id),
            "log_path": str(output_root / "workers" / f"{worker_id}.log"),
        }

    manifest = {
        "run_id": output_root.name,
        "run_slug": run_slug,
        "workspace_root": str(ROOT_DIR),
        "output_root": str(output_root),
        "state": "created",
        "created_at": now_iso(),
        "config": {
            "model": args.model,
            "model_spec": model_spec,
            "gpu_ids": gpu_ids,
            "attn_implementation": args.attn_implementation,
            "max_new_tokens": args.max_new_tokens,
            "chunk_size": args.chunk_size,
            "num_local_attention_blocks": args.num_local_attention_blocks,
            "retry_limit": args.retry_limit,
            "image": args.image,
            "hf_cache": str(hf_cache),
            "uv_cache": str(uv_cache),
            "skip_build": bool(args.skip_build),
            "data_root": str(ROOT_DIR / "datahub"),
        },
        "coordinator": {
            "pid": None,
            "status": "not_started",
            "log_path": str(output_root / "coordinator.log"),
            "started_at": None,
            "finished_at": None,
        },
        "workers": workers,
        "datasets": {},
        "chunks": [],
    }
    save_manifest(manifest_path, manifest)
    return manifest_path


def update_manifest_for_resume(manifest_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if args.gpu_ids:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        manifest["config"]["gpu_ids"] = gpu_ids
        run_slug = manifest["run_slug"]
        workers = {}
        for gpu_id in gpu_ids:
            worker_id = f"gpu{gpu_id}"
            workers[worker_id] = {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "container_name": worker_container_name(run_slug=run_slug, gpu_id=gpu_id),
                "log_path": str(Path(manifest["output_root"]) / "workers" / f"{worker_id}.log"),
            }
        manifest["workers"] = workers
    if args.attn_implementation:
        manifest["config"]["attn_implementation"] = args.attn_implementation
    if args.max_new_tokens is not None:
        manifest["config"]["max_new_tokens"] = args.max_new_tokens
    if args.chunk_size is not None:
        manifest["config"]["chunk_size"] = args.chunk_size
    if args.num_local_attention_blocks is not None:
        manifest["config"]["num_local_attention_blocks"] = args.num_local_attention_blocks
    if args.retry_limit is not None:
        manifest["config"]["retry_limit"] = args.retry_limit
    if args.image:
        manifest["config"]["image"] = args.image
    if args.hf_cache:
        manifest["config"]["hf_cache"] = str(Path(args.hf_cache).resolve())
    if args.uv_cache:
        manifest["config"]["uv_cache"] = str(Path(args.uv_cache).resolve())
    if args.skip_build:
        manifest["config"]["skip_build"] = True

    manifest["state"] = "created"
    manifest["coordinator"]["pid"] = None
    manifest["coordinator"]["status"] = "not_started"
    manifest["coordinator"]["finished_at"] = None
    save_manifest(manifest_path, manifest)
    return manifest


def detach_coordinator(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    coordinator_log = Path(manifest["coordinator"]["log_path"])
    coordinator_log.parent.mkdir(parents=True, exist_ok=True)

    with coordinator_log.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "table1_sharded.py"),
                "_run_coordinator",
                "--manifest",
                str(manifest_path),
            ],
            cwd=ROOT_DIR,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    manifest["coordinator"]["pid"] = process.pid
    manifest["coordinator"]["status"] = "starting"
    manifest["coordinator"]["started_at"] = now_iso()
    save_manifest(manifest_path, manifest)

    output_root = Path(manifest["output_root"])
    print(f"Run id: {manifest['run_id']}")
    print(f"Output root: {output_root}")
    print(f"Coordinator log: {coordinator_log}")
    print(f"Status: python3 scripts/table1_sharded.py status --output-root {output_root}")
    print(f"Stop: python3 scripts/table1_sharded.py stop --output-root {output_root}")


def parse_launch_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    subparser.add_argument("--gpu-ids", default="0,1,2,3")
    subparser.add_argument("--attn-implementation", default="eager")
    subparser.add_argument("--max-new-tokens", type=int, default=64)
    subparser.add_argument("--output-root", default=None)
    subparser.add_argument("--chunk-size", type=int, default=128)
    subparser.add_argument("--num-local-attention-blocks", type=int, default=10000)
    subparser.add_argument("--retry-limit", type=int, default=1)
    subparser.add_argument("--image", default="block-attention:latest")
    subparser.add_argument("--hf-cache", default=None)
    subparser.add_argument("--uv-cache", default=None)
    subparser.add_argument("--skip-build", action="store_true")


def parse_resume_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--output-root", required=True)
    subparser.add_argument("--gpu-ids", default=None)
    subparser.add_argument("--attn-implementation", default=None)
    subparser.add_argument("--max-new-tokens", type=int, default=None)
    subparser.add_argument("--chunk-size", type=int, default=None)
    subparser.add_argument("--num-local-attention-blocks", type=int, default=None)
    subparser.add_argument("--retry-limit", type=int, default=None)
    subparser.add_argument("--image", default=None)
    subparser.add_argument("--hf-cache", default=None)
    subparser.add_argument("--uv-cache", default=None)
    subparser.add_argument("--skip-build", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_launch_args(subparsers.add_parser("launch"))
    parse_resume_args(subparsers.add_parser("resume"))

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output-root", required=True)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--output-root", required=True)

    coordinator_parser = subparsers.add_parser("_run_coordinator")
    coordinator_parser.add_argument("--manifest", required=True)

    return parser.parse_args()


def chunk_sort_key(chunk: dict[str, Any]) -> tuple[int, int]:
    return (DATASET_ORDER.index(chunk["dataset"]), chunk["start_index"])


def prepare_container_command(manifest: dict[str, Any]) -> str:
    config = manifest["config"]
    model_spec = config["model_spec"]
    install_flash_attn = "1" if config["attn_implementation"] in {"auto", "flash_attention_2"} else "0"
    commands = [
        "set -euo pipefail",
        "cd /workspace",
        f"export INSTALL_FLASH_ATTN={install_flash_attn}",
        "bash scripts/prepare_table1_rag_eval.sh --data-root /workspace/datahub --venv /workspace/.venv --cuda-device cuda:0",
    ]

    if model_spec["mode"] == "hf_repo":
        repo_id = json.dumps(model_spec["repo_id"])
        commands.append(
            "/workspace/.venv/bin/python -c "
            + shlex.quote(
                "from huggingface_hub import snapshot_download; "
                "from pathlib import Path; "
                f"repo_id = {repo_id}; "
                "target = Path('/workspace/models') / Path(repo_id).name; "
                "target.mkdir(parents=True, exist_ok=True); "
                "print(snapshot_download(repo_id=repo_id, local_dir=str(target)))"
            )
        )
    elif model_spec["mode"] == "external_path":
        commands.append("test -d /mounted_model")
    else:
        commands.append(f"test -d {shlex.quote(model_spec['container_path'])}")

    return "\n".join(commands)


def one_off_container(
    *,
    manifest: dict[str, Any],
    name: str,
    gpu_id: str,
    command: str,
) -> None:
    config = manifest["config"]
    model_spec = config["model_spec"]
    extra_mounts: list[tuple[Path, str, bool]] = []
    if model_spec["mode"] == "external_path":
        extra_mounts.append((Path(model_spec["host_path"]), model_spec["container_path"], True))

    docker_remove_container(name)
    docker(
        [
            "run",
            "--rm",
            "--name",
            name,
            "--gpus",
            f"device={gpu_id}",
            *host_mount_env(
                hf_cache=Path(config["hf_cache"]),
                uv_cache=Path(config["uv_cache"]),
                extra_mounts=extra_mounts,
            ),
            *container_env_args(),
            config["image"],
            "bash",
            "-lc",
            command,
        ]
    )


def initialize_chunks(manifest: dict[str, Any], manifest_path: Path) -> None:
    if manifest["chunks"]:
        return

    data_root = Path(manifest["config"]["data_root"])
    chunk_size = int(manifest["config"]["chunk_size"])
    output_root = Path(manifest["output_root"])

    chunks: list[dict[str, Any]] = []
    datasets = {}
    for dataset_name in DATASET_ORDER:
        input_path = data_root / "rag" / f"{dataset_name}_eval" / "dataset"
        count = count_jsonl_lines(input_path)
        datasets[dataset_name] = {
            "input_path": str(input_path),
            "count": count,
        }

        dataset_chunk_dir = output_root / "chunks" / dataset_name
        dataset_chunk_dir.mkdir(parents=True, exist_ok=True)
        for chunk_index, start_index in enumerate(range(0, count, chunk_size)):
            end_index = min(start_index + chunk_size, count)
            chunk_output = dataset_chunk_dir / f"chunk_{chunk_index:04d}.jsonl"
            chunk_done = dataset_chunk_dir / f"chunk_{chunk_index:04d}.done"
            chunk_log = dataset_chunk_dir / f"chunk_{chunk_index:04d}.log"
            chunks.append(
                {
                    "id": f"{dataset_name}-{chunk_index:04d}",
                    "dataset": dataset_name,
                    "chunk_index": chunk_index,
                    "start_index": start_index,
                    "end_index": end_index,
                    "expected_count": end_index - start_index,
                    "output_path": str(chunk_output),
                    "done_marker": str(chunk_done),
                    "log_path": str(chunk_log),
                    "status": "pending",
                    "attempts": 0,
                    "assigned_worker": None,
                    "last_error": "",
                    "updated_at": now_iso(),
                }
            )

    manifest["datasets"] = datasets
    manifest["chunks"] = chunks
    save_manifest(manifest_path, manifest)


def reconcile_chunks(manifest: dict[str, Any], manifest_path: Path) -> None:
    changed = False
    for chunk in manifest["chunks"]:
        output_path = Path(chunk["output_path"])
        done_marker = Path(chunk["done_marker"])
        count = count_recovered_output(output_path)

        if count > chunk["expected_count"]:
            raise ValueError(
                f"Chunk output {output_path} has {count} rows, expected at most {chunk['expected_count']}"
            )

        if done_marker.exists() and count == chunk["expected_count"]:
            if chunk["status"] != "done":
                chunk["status"] = "done"
                chunk["assigned_worker"] = None
                chunk["last_error"] = ""
                chunk["updated_at"] = now_iso()
                changed = True
            continue

        if chunk["status"] in {"running", "failed"} or count < chunk["expected_count"]:
            if chunk["status"] != "pending":
                chunk["status"] = "pending"
                chunk["assigned_worker"] = None
                chunk["updated_at"] = now_iso()
                changed = True

    if changed:
        save_manifest(manifest_path, manifest)


def start_worker_container(manifest: dict[str, Any], worker: dict[str, Any]) -> None:
    config = manifest["config"]
    model_spec = config["model_spec"]
    worker_log_path = Path(worker["log_path"])
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_log_container_path = to_container_workspace_path(worker_log_path)
    extra_mounts: list[tuple[Path, str, bool]] = []
    if model_spec["mode"] == "external_path":
        extra_mounts.append((Path(model_spec["host_path"]), model_spec["container_path"], True))

    docker_remove_container(worker["container_name"])

    server_command = "\n".join(
        [
            "set -euo pipefail",
            "cd /workspace",
            f"mkdir -p {shlex.quote(str(Path(worker_log_container_path).parent))}",
            (
                "exec /workspace/.venv/bin/python server/block_generate_server.py "
                f"--model {shlex.quote(model_spec['container_path'])} "
                "--port 8080 "
                "--dtype bfloat16 "
                f"--attn_implementation {shlex.quote(config['attn_implementation'])} "
                f"--max_new_tokens {int(config['max_new_tokens'])} "
                f">>{shlex.quote(worker_log_container_path)} 2>&1"
            ),
        ]
    )

    docker(
        [
            "run",
            "-d",
            "--name",
            worker["container_name"],
            "--gpus",
            f"device={worker['gpu_id']}",
            *host_mount_env(
                hf_cache=Path(config["hf_cache"]),
                uv_cache=Path(config["uv_cache"]),
                extra_mounts=extra_mounts,
            ),
            *container_env_args(),
            config["image"],
            "bash",
            "-lc",
            server_command,
        ]
    )


def wait_for_worker_health(worker: dict[str, Any], timeout_seconds: int = 1800) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        if not docker_container_running(worker["container_name"]):
            raise RuntimeError(f"Worker container exited early: {worker['container_name']}")

        health_check = worker_health_check(worker)
        if health_check.returncode == 0:
            return
        time.sleep(10)

    raise TimeoutError(f"Worker did not become healthy: {worker['container_name']}")


def worker_health_check(worker: dict[str, Any]) -> subprocess.CompletedProcess[Any]:
    return docker(
        [
            "exec",
            worker["container_name"],
            "/workspace/.venv/bin/python",
            "-c",
            (
                "import requests; "
                "response = requests.post("
                "'http://127.0.0.1:8080/generate', "
                "json={'blocks': ['<|user|>\\nYou are an intelligent AI assistant. Please answer questions based on the user\\'s instructions. Below are some reference documents that may help you in answering the user\\'s question.\\n\\n', '- Title: Warmup\\nWarmup\\n', '\\n\\nPlease write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant).\\nQuestion: What is the title?\\n<|assistant|>\\n']}, "
                "timeout=10); "
                "response.raise_for_status()"
            ),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=False,
    )


def ensure_workers_ready(manifest: dict[str, Any]) -> None:
    for worker in manifest["workers"].values():
        start_worker_container(manifest, worker)
    for worker in manifest["workers"].values():
        wait_for_worker_health(worker)


def next_pending_chunk(manifest: dict[str, Any]) -> dict[str, Any] | None:
    pending_chunks = [chunk for chunk in manifest["chunks"] if chunk["status"] == "pending"]
    if not pending_chunks:
        return None
    pending_chunks.sort(key=chunk_sort_key)
    return pending_chunks[0]


def start_chunk_process(worker: dict[str, Any], chunk: dict[str, Any], manifest: dict[str, Any]) -> ActiveChunkProcess:
    config = manifest["config"]
    input_path = Path(manifest["datasets"][chunk["dataset"]]["input_path"])
    chunk_log_path = Path(chunk["log_path"])
    chunk_log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "docker",
        "exec",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        worker["container_name"],
        "/workspace/.venv/bin/python",
        "/workspace/scripts/run_rag_block_inference.py",
        "--input",
        to_container_workspace_path(input_path),
        "--output",
        to_container_workspace_path(Path(chunk["output_path"])),
        "--done-marker",
        to_container_workspace_path(Path(chunk["done_marker"])),
        "--server-url",
        "http://127.0.0.1:8080/generate",
        "--num-local-attention-blocks",
        str(config["num_local_attention_blocks"]),
        "--start-index",
        str(chunk["start_index"]),
        "--end-index",
        str(chunk["end_index"]),
    ]

    log_handle = chunk_log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=ROOT_DIR,
    )
    return ActiveChunkProcess(
        worker_id=worker["worker_id"],
        chunk_id=chunk["id"],
        process=process,
        log_handle=log_handle,
    )


def mark_chunk_running(manifest: dict[str, Any], manifest_path: Path, chunk: dict[str, Any], worker_id: str) -> None:
    chunk["status"] = "running"
    chunk["assigned_worker"] = worker_id
    chunk["attempts"] += 1
    chunk["last_error"] = ""
    chunk["updated_at"] = now_iso()
    save_manifest(manifest_path, manifest)


def mark_chunk_done(manifest: dict[str, Any], manifest_path: Path, chunk: dict[str, Any]) -> None:
    chunk["status"] = "done"
    chunk["assigned_worker"] = None
    chunk["last_error"] = ""
    chunk["updated_at"] = now_iso()
    save_manifest(manifest_path, manifest)


def mark_chunk_failed_or_pending(
    manifest: dict[str, Any],
    manifest_path: Path,
    chunk: dict[str, Any],
    error_message: str,
) -> None:
    max_attempts = int(manifest["config"]["retry_limit"]) + 1
    chunk["assigned_worker"] = None
    chunk["last_error"] = error_message
    chunk["updated_at"] = now_iso()
    if chunk["attempts"] < max_attempts:
        chunk["status"] = "pending"
    else:
        chunk["status"] = "failed"
    save_manifest(manifest_path, manifest)


def stop_workers(manifest: dict[str, Any]) -> None:
    for worker in manifest["workers"].values():
        docker_remove_container(worker["container_name"])


def merge_dataset_outputs(manifest: dict[str, Any]) -> None:
    output_root = Path(manifest["output_root"])
    generated_dir = output_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASET_ORDER}
    for chunk in manifest["chunks"]:
        if chunk["status"] != "done":
            raise RuntimeError(f"Cannot merge while chunk is not done: {chunk['id']}")
        chunks_by_dataset[chunk["dataset"]].append(chunk)

    for dataset_name in DATASET_ORDER:
        chunks = sorted(chunks_by_dataset[dataset_name], key=lambda chunk: chunk["start_index"])
        output_path = generated_dir / f"{dataset_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as output_handle:
            merged_count = 0
            for chunk in chunks:
                chunk_path = Path(chunk["output_path"])
                records = read_jsonl_with_recovery(chunk_path)
                if len(records) != chunk["expected_count"]:
                    raise RuntimeError(
                        f"Chunk {chunk['id']} has {len(records)} rows, expected {chunk['expected_count']}"
                    )
                for record in records:
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                merged_count += len(records)

        expected_count = manifest["datasets"][dataset_name]["count"]
        if merged_count != expected_count:
            raise RuntimeError(
                f"Merged {dataset_name} count mismatch: expected {expected_count}, got {merged_count}"
            )


def write_summary_via_worker(manifest: dict[str, Any]) -> None:
    first_worker = next(iter(manifest["workers"].values()))
    output_root = Path(manifest["output_root"])
    generated_dir = output_root / "generated"
    command = [
        "exec",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        first_worker["container_name"],
        "/workspace/.venv/bin/python",
        "/workspace/scripts/write_table1_summary.py",
        "--output-prefix",
        to_container_workspace_path(output_root / "results"),
        "--2wiki",
        to_container_workspace_path(generated_dir / "2wiki.jsonl"),
        "--hqa",
        to_container_workspace_path(generated_dir / "hqa.jsonl"),
        "--nq",
        to_container_workspace_path(generated_dir / "nq.jsonl"),
        "--tqa",
        to_container_workspace_path(generated_dir / "tqa.jsonl"),
    ]
    docker(command)


def summarize_status(manifest: dict[str, Any]) -> str:
    by_status: dict[str, int] = {}
    for chunk in manifest["chunks"]:
        by_status[chunk["status"]] = by_status.get(chunk["status"], 0) + 1
    status_parts = [f"{key}={by_status.get(key, 0)}" for key in ["pending", "running", "done", "failed"]]
    return " ".join(status_parts)


def coordinator_main(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    manifest["state"] = "running"
    manifest["coordinator"]["pid"] = os.getpid()
    manifest["coordinator"]["status"] = "running"
    manifest["coordinator"]["started_at"] = manifest["coordinator"]["started_at"] or now_iso()
    save_manifest(manifest_path, manifest)

    print(f"[{now_iso()}] Starting sharded Table 1 run: {manifest['output_root']}", flush=True)

    if not manifest["config"]["skip_build"]:
        print(f"[{now_iso()}] Building docker image: {manifest['config']['image']}", flush=True)
        docker_build(manifest["config"]["image"])

    print(f"[{now_iso()}] Preparing datasets, venv, and model cache", flush=True)
    one_off_container(
        manifest=manifest,
        name=prep_container_name(manifest["run_slug"]),
        gpu_id=manifest["config"]["gpu_ids"][0],
        command=prepare_container_command(manifest),
    )

    initialize_chunks(manifest, manifest_path)
    manifest = load_manifest(manifest_path)
    reconcile_chunks(manifest, manifest_path)
    manifest = load_manifest(manifest_path)

    print(f"[{now_iso()}] Starting worker containers", flush=True)
    stop_workers(manifest)
    ensure_workers_ready(manifest)
    print(f"[{now_iso()}] Workers are healthy. {summarize_status(manifest)}", flush=True)

    active_processes: dict[str, ActiveChunkProcess] = {}

    try:
        while True:
            manifest = load_manifest(manifest_path)

            for worker_id, active in list(active_processes.items()):
                exit_code = active.process.poll()
                if exit_code is None:
                    continue

                active.log_handle.close()
                del active_processes[worker_id]

                manifest = load_manifest(manifest_path)
                chunk = next(item for item in manifest["chunks"] if item["id"] == active.chunk_id)
                worker = manifest["workers"][worker_id]
                output_count = count_recovered_output(Path(chunk["output_path"]))
                done_marker_exists = Path(chunk["done_marker"]).exists()

                if exit_code == 0 and done_marker_exists and output_count == chunk["expected_count"]:
                    mark_chunk_done(manifest, manifest_path, chunk)
                    print(f"[{now_iso()}] Completed {chunk['id']} on {worker_id}", flush=True)
                else:
                    error_message = (
                        f"Chunk {chunk['id']} failed on {worker_id} with exit code {exit_code}. "
                        f"Recovered {output_count}/{chunk['expected_count']} rows."
                    )
                    mark_chunk_failed_or_pending(manifest, manifest_path, chunk, error_message)
                    print(f"[{now_iso()}] {error_message}", flush=True)
                    if not docker_container_running(worker["container_name"]) or worker_health_check(worker).returncode != 0:
                        print(f"[{now_iso()}] Restarting worker container {worker['container_name']}", flush=True)
                        start_worker_container(manifest, worker)
                        wait_for_worker_health(worker)

            manifest = load_manifest(manifest_path)
            failed_chunks = [chunk for chunk in manifest["chunks"] if chunk["status"] == "failed"]
            if failed_chunks:
                raise RuntimeError(f"One or more chunks exceeded retry limit: {[chunk['id'] for chunk in failed_chunks]}")

            if all(chunk["status"] == "done" for chunk in manifest["chunks"]):
                print(f"[{now_iso()}] All chunks finished. Merging outputs.", flush=True)
                merge_dataset_outputs(manifest)
                write_summary_via_worker(manifest)
                manifest["state"] = "completed"
                manifest["coordinator"]["status"] = "completed"
                manifest["coordinator"]["finished_at"] = now_iso()
                save_manifest(manifest_path, manifest)
                print(f"[{now_iso()}] Finished. Results: {Path(manifest['output_root']) / 'results.md'}", flush=True)
                return

            for worker_id, worker in manifest["workers"].items():
                if worker_id in active_processes:
                    continue

                if not docker_container_running(worker["container_name"]):
                    print(f"[{now_iso()}] Worker {worker_id} is down. Restarting.", flush=True)
                    start_worker_container(manifest, worker)
                    wait_for_worker_health(worker)
                elif worker_health_check(worker).returncode != 0:
                    print(f"[{now_iso()}] Worker {worker_id} is unhealthy. Restarting.", flush=True)
                    start_worker_container(manifest, worker)
                    wait_for_worker_health(worker)

                chunk = next_pending_chunk(manifest)
                if chunk is None:
                    break

                mark_chunk_running(manifest, manifest_path, chunk, worker_id)
                active_process = start_chunk_process(worker, chunk, manifest)
                active_processes[worker_id] = active_process
                print(
                    f"[{now_iso()}] Started {chunk['id']} on {worker_id} "
                    f"({chunk['start_index']}:{chunk['end_index']})",
                    flush=True,
                )
                manifest = load_manifest(manifest_path)

            if active_processes:
                time.sleep(5)
            else:
                time.sleep(2)
    except Exception as exc:
        manifest = load_manifest(manifest_path)
        manifest["state"] = "failed"
        manifest["coordinator"]["status"] = "failed"
        manifest["coordinator"]["finished_at"] = now_iso()
        save_manifest(manifest_path, manifest)
        print(f"[{now_iso()}] Coordinator failed: {exc}", flush=True)
        raise
    finally:
        for active in active_processes.values():
            if active.process.poll() is None:
                active.process.terminate()
                try:
                    active.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    active.process.kill()
            active.log_handle.close()
        stop_workers(load_manifest(manifest_path))


def status_command(output_root: str) -> None:
    manifest_path = Path(output_root).resolve() / "manifest.json"
    manifest = load_manifest(manifest_path)
    coordinator_pid = manifest["coordinator"]["pid"]
    print(f"Run id: {manifest['run_id']}")
    print(f"State: {manifest['state']}")
    print(f"Coordinator pid: {coordinator_pid}")
    print(f"Coordinator alive: {is_process_alive(coordinator_pid)}")
    print(f"Coordinator log: {manifest['coordinator']['log_path']}")
    print(f"Chunk status: {summarize_status(manifest)}")
    for worker in manifest["workers"].values():
        print(
            f"Worker {worker['worker_id']} gpu={worker['gpu_id']} "
            f"container={worker['container_name']} running={docker_container_running(worker['container_name'])}"
        )


def stop_command(output_root: str) -> None:
    manifest_path = Path(output_root).resolve() / "manifest.json"
    manifest = load_manifest(manifest_path)
    coordinator_pid = manifest["coordinator"]["pid"]
    if is_process_alive(coordinator_pid):
        os.kill(coordinator_pid, signal.SIGTERM)
        time.sleep(1)
        if is_process_alive(coordinator_pid):
            os.kill(coordinator_pid, signal.SIGKILL)
    stop_workers(manifest)
    manifest["state"] = "stopped"
    manifest["coordinator"]["status"] = "stopped"
    manifest["coordinator"]["finished_at"] = now_iso()
    save_manifest(manifest_path, manifest)
    print(f"Stopped run: {manifest['run_id']}")


def main() -> None:
    args = parse_args()

    if args.command == "launch":
        manifest_path = create_manifest(args)
        detach_coordinator(manifest_path)
        return

    if args.command == "resume":
        manifest_path = Path(args.output_root).resolve() / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        manifest = load_manifest(manifest_path)
        if is_process_alive(manifest["coordinator"]["pid"]):
            raise RuntimeError(f"Coordinator is already running for {manifest['run_id']}")
        update_manifest_for_resume(manifest_path, args)
        detach_coordinator(manifest_path)
        return

    if args.command == "status":
        status_command(args.output_root)
        return

    if args.command == "stop":
        stop_command(args.output_root)
        return

    if args.command == "_run_coordinator":
        coordinator_main(Path(args.manifest).resolve())
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
