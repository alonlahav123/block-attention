import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from requests import Response
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_prompting import build_rag_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--num-local-attention-blocks", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--done-marker", default=None)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def recover_output_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    recovered_examples: list[dict[str, Any]] = []
    last_good_offset = 0
    file_size = path.stat().st_size

    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break

            try:
                decoded = line.decode("utf-8")
                recovered_examples.append(json.loads(decoded))
                last_good_offset = handle.tell()
            except Exception:
                if handle.tell() != file_size:
                    raise ValueError(
                        f"Encountered malformed JSON before EOF in output file: {path}"
                    )
                with path.open("rb+") as writable_handle:
                    writable_handle.truncate(line_start)
                break

    return recovered_examples


def iter_slice(items: list[dict[str, Any]], start_index: int, end_index: int) -> Iterable[dict[str, Any]]:
    return items[start_index:end_index]


def append_example(path: Path, example: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_request_payload(example: dict[str, Any], num_local_attention_blocks: int) -> dict[str, Any]:
    return {
        "blocks": build_rag_blocks(
            question=example["question"],
            documents=example["documents"],
        ),
        "num_local_attention_blocks": num_local_attention_blocks,
    }


def post_generate(
    server_url: str,
    payload: dict[str, Any],
    request_timeout: int,
) -> Response:
    response = requests.post(server_url, json=payload, timeout=request_timeout)
    response.raise_for_status()
    return response


def get_response(
    server_url: str,
    payload: dict[str, Any],
    request_timeout: int,
    max_retries: int,
    retry_sleep_seconds: float,
    question: str,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = post_generate(
                server_url=server_url,
                payload=payload,
                request_timeout=request_timeout,
            )
            body = response.json()
            if body.get("ret") != 0:
                raise RuntimeError(body.get("message", "Server returned a failure response"))
            return body["generated"]
        except Exception as exc:
            last_error = exc
            print(
                f"Request failed on attempt {attempt}/{max_retries} for question: {question}",
                flush=True,
            )
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"Failed after {max_retries} attempts") from last_error


def write_done_marker(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps({"count": count}, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    done_marker = Path(args.done_marker).resolve() if args.done_marker else None

    all_examples = load_jsonl(path=input_path)
    end_index = args.end_index if args.end_index is not None else len(all_examples)
    if not (0 <= args.start_index <= end_index <= len(all_examples)):
        raise ValueError(
            f"Invalid slice [{args.start_index}, {end_index}) for dataset of size {len(all_examples)}"
        )

    selected_examples = list(iter_slice(all_examples, args.start_index, end_index))
    existing_examples = recover_output_file(path=output_path)

    if len(existing_examples) > len(selected_examples):
        raise ValueError(
            f"Output file {output_path} already contains {len(existing_examples)} examples, "
            f"but the requested slice only has {len(selected_examples)}"
        )

    if done_marker is not None and done_marker.exists() and len(existing_examples) == len(selected_examples):
        return

    progress_bar = tqdm(
        total=len(selected_examples),
        initial=len(existing_examples),
        desc="Generate RAG",
    )

    try:
        for example in selected_examples[len(existing_examples):]:
            payload = build_request_payload(
                example=example,
                num_local_attention_blocks=args.num_local_attention_blocks,
            )
            example["generated"] = get_response(
                server_url=args.server_url,
                payload=payload,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
                question=example["question"],
            )
            append_example(path=output_path, example=example)
            progress_bar.update(1)
    finally:
        progress_bar.close()

    final_examples = recover_output_file(path=output_path)
    if len(final_examples) != len(selected_examples):
        raise RuntimeError(
            f"Expected {len(selected_examples)} generated examples in {output_path}, "
            f"found {len(final_examples)}"
        )

    if done_marker is not None:
        write_done_marker(path=done_marker, count=len(final_examples))


if __name__ == "__main__":
    main()
