import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests
from requests import Response
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag_prompting import build_rag_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--done-marker", default=None)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--request-concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--metrics-output", default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def recover_output_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    recovered_examples: list[dict[str, Any]] = []
    file_size = path.stat().st_size

    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break

            try:
                recovered_examples.append(json.loads(line.decode("utf-8")))
            except Exception:
                if handle.tell() != file_size:
                    raise ValueError(
                        f"Encountered malformed JSON before EOF in output file: {path}"
                    )
                with path.open("rb+") as writable_handle:
                    writable_handle.truncate(line_start)
                break

    return recovered_examples


def append_example(path: Path, example: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_done_marker(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps({"count": count}, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def write_metrics(path: Path, rows: list[dict[str, Any]], wall_seconds: float) -> None:
    latencies = [row["request_latency_seconds"] for row in rows if "request_latency_seconds" in row]
    payload = {
        "count": len(rows),
        "wall_seconds": wall_seconds,
        "examples_per_second": (len(rows) / wall_seconds) if wall_seconds > 0 else 0.0,
        "median_request_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "max_request_latency_seconds": max(latencies) if latencies else 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_generated_text(body: dict[str, Any], prompt: str) -> str:
    generated = body.get("generated")
    if isinstance(generated, str):
        return generated

    text = body.get("text")
    if isinstance(text, str):
        return text[len(prompt):] if text.startswith(prompt) else text

    if isinstance(text, list) and text:
        first = text[0]
        if isinstance(first, str):
            return first[len(prompt):] if first.startswith(prompt) else first

    raise RuntimeError(f"Unsupported response payload keys: {sorted(body.keys())}")


def post_generate(server_url: str, payload: dict[str, Any], request_timeout: int) -> Response:
    response = requests.post(server_url, json=payload, timeout=request_timeout)
    response.raise_for_status()
    return response


def generate_example(
    *,
    server_url: str,
    question: str,
    documents: list[dict[str, Any]],
    request_timeout: int,
    max_retries: int,
    retry_sleep_seconds: float,
    max_new_tokens: int,
) -> tuple[str, float]:
    prompt = build_rag_prompt(question=question, documents=documents)
    payload = {
        "prompt": prompt,
        "max_tokens": max_new_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        start_time = time.perf_counter()
        try:
            response = post_generate(
                server_url=server_url,
                payload=payload,
                request_timeout=request_timeout,
            )
            latency = time.perf_counter() - start_time
            body = response.json()
            return parse_generated_text(body=body, prompt=prompt), latency
        except Exception as exc:
            last_error = exc
            print(
                f"Request failed on attempt {attempt}/{max_retries} for question: {question}",
                flush=True,
            )
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"Failed after {max_retries} attempts") from last_error


def validate_existing_rows(
    rows: list[dict[str, Any]],
    *,
    start_index: int,
    end_index: int,
) -> set[int]:
    seen_indexes: set[int] = set()
    for row in rows:
        if "example_index" not in row:
            raise ValueError("Recovered row is missing 'example_index'")
        example_index = int(row["example_index"])
        if example_index < start_index or example_index >= end_index:
            raise ValueError(
                f"Recovered example_index {example_index} falls outside requested slice [{start_index}, {end_index})"
            )
        if example_index in seen_indexes:
            raise ValueError(f"Duplicate example_index recovered: {example_index}")
        seen_indexes.add(example_index)
    return seen_indexes


def build_completed_rows_by_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["example_index"]): row for row in rows}


def main() -> None:
    args = parse_args()

    if args.request_concurrency < 1:
        raise ValueError("--request-concurrency must be at least 1")

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    done_marker = Path(args.done_marker).resolve() if args.done_marker else None
    metrics_output = Path(args.metrics_output).resolve() if args.metrics_output else None

    all_examples = load_jsonl(path=input_path)
    end_index = args.end_index if args.end_index is not None else len(all_examples)
    if not (0 <= args.start_index <= end_index <= len(all_examples)):
        raise ValueError(
            f"Invalid slice [{args.start_index}, {end_index}) for dataset of size {len(all_examples)}"
        )

    selected_examples = all_examples[args.start_index:end_index]
    existing_rows = recover_output_file(path=output_path)
    completed_indexes = validate_existing_rows(
        existing_rows,
        start_index=args.start_index,
        end_index=end_index,
    )

    if done_marker is not None and done_marker.exists() and len(completed_indexes) == len(selected_examples):
        return

    pending_items: list[tuple[int, dict[str, Any]]] = []
    for relative_index, example in enumerate(selected_examples):
        example_index = args.start_index + relative_index
        if example_index not in completed_indexes:
            pending_items.append((example_index, example))

    progress_bar = tqdm(
        total=len(selected_examples),
        initial=len(completed_indexes),
        desc="Generate RAG",
    )

    start_wall_time = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.request_concurrency) as executor:
            futures = {}
            pending_iter = iter(pending_items)

            def submit_more() -> None:
                while len(futures) < args.request_concurrency:
                    try:
                        example_index, example = next(pending_iter)
                    except StopIteration:
                        return
                    futures[
                        executor.submit(
                            generate_example,
                            server_url=args.server_url,
                            question=example["question"],
                            documents=example["documents"],
                            request_timeout=args.request_timeout,
                            max_retries=args.max_retries,
                            retry_sleep_seconds=args.retry_sleep_seconds,
                            max_new_tokens=args.max_new_tokens,
                        )
                    ] = (example_index, example)

            submit_more()
            while futures:
                done_futures, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for future in done_futures:
                    example_index, example = futures.pop(future)
                    generated, latency = future.result()
                    output_row = dict(example)
                    output_row["example_index"] = example_index
                    output_row["generated"] = generated
                    output_row["request_latency_seconds"] = latency
                    append_example(path=output_path, example=output_row)
                    progress_bar.update(1)
                submit_more()
    finally:
        progress_bar.close()

    final_rows = recover_output_file(path=output_path)
    final_completed_indexes = validate_existing_rows(
        final_rows,
        start_index=args.start_index,
        end_index=end_index,
    )
    if len(final_completed_indexes) != len(selected_examples):
        raise RuntimeError(
            f"Expected {len(selected_examples)} generated examples in {output_path}, "
            f"found {len(final_completed_indexes)}"
        )

    if done_marker is not None:
        write_done_marker(path=done_marker, count=len(final_rows))
    if metrics_output is not None:
        selected_rows = [
            row for _, row in sorted(build_completed_rows_by_index(final_rows).items())
            if args.start_index <= int(row["example_index"]) < end_index
        ]
        write_metrics(metrics_output, selected_rows, wall_seconds=time.perf_counter() - start_wall_time)


if __name__ == "__main__":
    main()
