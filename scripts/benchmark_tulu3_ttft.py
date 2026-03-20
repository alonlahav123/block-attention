import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.block_generation import (
    build_default_generation_config,
    build_rotary_embedding,
    count_block_prompt_tokens,
    decode_generated_tokens,
    encode_block_inputs,
    generate_block_tokens,
)
from src.rag_prompting import build_rag_blocks, build_rag_prompt

DATASET_NAMES = ["2wiki", "hqa", "nq", "tqa"]
DATASET_PATHS = {
    "2wiki": "rag/2wiki_eval/dataset",
    "hqa": "rag/hqa_eval/dataset",
    "nq": "rag/nq_eval/dataset",
    "tqa": "rag/tqa_eval/dataset",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-model", default="ldsjmdy/Tulu3-RAG")
    parser.add_argument("--block-model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument("--data-root", default=str(ROOT_DIR / "datahub"))
    parser.add_argument("--output-root", default=str(ROOT_DIR / "outputs" / "tulu3_ttft"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--samples-per-dataset", type=int, default=16)
    parser.add_argument("--warmup-per-dataset", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument(
        "--prepare-data",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--internal-mode",
        choices=["probe-attention", "run-model"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--model-kind", choices=["rag", "block"], help=argparse.SUPPRESS)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--subset-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--per-example-output", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_prepare_data(data_root: Path) -> None:
    cmd = [
        "bash",
        str(ROOT_DIR / "scripts" / "prepare_table1_rag_eval.sh"),
        "--data-root",
        str(data_root),
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def resolve_model_path(model_source: str) -> Path:
    candidate = Path(model_source).expanduser()
    if candidate.exists():
        return candidate.resolve()

    model_cache_dir = ROOT_DIR / "models" / Path(model_source).name
    model_cache_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=model_source,
            local_dir=str(model_cache_dir),
        )
    ).resolve()


def dataset_path(data_root: Path, dataset_name: str) -> Path:
    return (data_root / DATASET_PATHS[dataset_name]).resolve()


def resolve_context_limit(config: AutoConfig) -> int:
    for attr_name in [
        "max_position_embeddings",
        "max_sequence_length",
        "max_model_len",
        "n_positions",
    ]:
        value = getattr(config, attr_name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Unable to determine model context limit from config")


def build_selection_manifest(
    *,
    rag_model_source: str,
    block_model_source: str,
    rag_model_path: Path,
    block_model_path: Path,
    data_root: Path,
    output_root: Path,
    seed: int,
    warmup_per_dataset: int,
    samples_per_dataset: int,
    attn_implementation: str,
) -> Path:
    rag_tokenizer = AutoTokenizer.from_pretrained(str(rag_model_path), use_fast=False)
    block_tokenizer = AutoTokenizer.from_pretrained(str(block_model_path), use_fast=False)
    rag_context_limit = resolve_context_limit(
        AutoConfig.from_pretrained(str(rag_model_path))
    )
    block_context_limit = resolve_context_limit(
        AutoConfig.from_pretrained(str(block_model_path))
    )

    rng = random.Random(seed)
    datasets_payload: dict[str, Any] = {}

    for dataset_name in DATASET_NAMES:
        examples = load_jsonl(dataset_path(data_root, dataset_name))
        eligible_entries: list[dict[str, Any]] = []

        for example_index, example in enumerate(examples):
            documents = example["documents"]
            question = example["question"]

            rag_prompt = build_rag_prompt(question=question, documents=documents)
            rag_prompt_tokens = len(
                rag_tokenizer.encode(rag_prompt, add_special_tokens=False)
            )

            block_parts = build_rag_blocks(question=question, documents=documents)
            encoded_block_inputs = encode_block_inputs(
                blocks=block_parts[:-1],
                instruction=block_parts[-1],
                tokenizer=block_tokenizer,
            )
            block_prompt_tokens = count_block_prompt_tokens(encoded_block_inputs)

            if rag_prompt_tokens + 1 > rag_context_limit:
                continue
            if block_prompt_tokens + 1 > block_context_limit:
                continue

            eligible_entries.append(
                {
                    "dataset": dataset_name,
                    "example_index": example_index,
                    "document_count": len(documents),
                    "rag_prompt_tokens": rag_prompt_tokens,
                    "block_prompt_tokens": block_prompt_tokens,
                    "num_local_attention_blocks": len(block_parts) - 1,
                }
            )

        required_examples = warmup_per_dataset + samples_per_dataset
        if len(eligible_entries) < required_examples:
            raise ValueError(
                f"Dataset {dataset_name} only has {len(eligible_entries)} eligible examples "
                f"after filtering, but {required_examples} are required."
            )

        rng.shuffle(eligible_entries)
        datasets_payload[dataset_name] = {
            "available_after_filtering": len(eligible_entries),
            "warmup": eligible_entries[:warmup_per_dataset],
            "measured": eligible_entries[
                warmup_per_dataset : warmup_per_dataset + samples_per_dataset
            ],
        }

    manifest_path = output_root / "subset_manifest.json"
    write_json(
        manifest_path,
        {
            "seed": seed,
            "warmup_per_dataset": warmup_per_dataset,
            "samples_per_dataset": samples_per_dataset,
            "shared_attn_implementation": attn_implementation,
            "data_root": str(data_root),
            "models": {
                "rag": {
                    "source": rag_model_source,
                    "path": str(rag_model_path),
                    "context_limit": rag_context_limit,
                },
                "block": {
                    "source": block_model_source,
                    "path": str(block_model_path),
                    "context_limit": block_context_limit,
                },
            },
            "datasets": datasets_payload,
        },
    )
    return manifest_path


def resolve_dtype() -> torch.dtype:
    return torch.bfloat16


def probe_attention_support(*, model_path: str, gpu_id: int, attn_implementation: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TTFT benchmarking")

    device = f"cuda:{gpu_id}"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=resolve_dtype(),
        device_map=device,
        attn_implementation=attn_implementation,
    )
    model.eval()
    del model
    gc_cuda()


def gc_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def subprocess_probe_attention(
    *,
    model_path: Path,
    gpu_id: int,
    attn_implementation: str,
) -> bool:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-mode",
        "probe-attention",
        "--model-path",
        str(model_path),
        "--gpu-id",
        str(gpu_id),
        "--attn-implementation",
        attn_implementation,
    ]
    completed = subprocess.run(cmd, cwd=ROOT_DIR, check=False)
    return completed.returncode == 0


def resolve_shared_attention(
    *,
    rag_model_path: Path,
    block_model_path: Path,
    gpu_id: int,
    requested_attn_implementation: str,
) -> str:
    if requested_attn_implementation == "auto":
        candidates = ["flash_attention_2", "sdpa"]
    else:
        candidates = [requested_attn_implementation]

    for candidate in candidates:
        if not subprocess_probe_attention(
            model_path=rag_model_path,
            gpu_id=gpu_id,
            attn_implementation=candidate,
        ):
            continue
        if not subprocess_probe_attention(
            model_path=block_model_path,
            gpu_id=gpu_id,
            attn_implementation=candidate,
        ):
            continue
        return candidate

    raise RuntimeError(
        "Could not find an attention implementation that works for both models"
    )


def run_model_subprocess(
    *,
    model_kind: str,
    model_path: Path,
    subset_manifest_path: Path,
    per_example_output_path: Path,
    gpu_id: int,
    attn_implementation: str,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-mode",
        "run-model",
        "--model-kind",
        model_kind,
        "--model-path",
        str(model_path),
        "--subset-manifest",
        str(subset_manifest_path),
        "--per-example-output",
        str(per_example_output_path),
        "--gpu-id",
        str(gpu_id),
        "--attn-implementation",
        attn_implementation,
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[index]


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator_x = sum((x - x_mean) ** 2 for x in xs)
    denominator_y = sum((y - y_mean) ** 2 for y in ys)
    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / math.sqrt(denominator_x * denominator_y)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured_rows = [row for row in rows if row["phase"] == "measured"]
    ttft_values = [row["ttft_seconds"] for row in measured_rows]
    prompt_tokens = [row["prompt_tokens"] for row in measured_rows]
    bucket_width = 1024
    if prompt_tokens and max(prompt_tokens) > 4096:
        bucket_width = 2048

    per_dataset: dict[str, Any] = {}
    for dataset_name in DATASET_NAMES:
        dataset_rows = [row for row in measured_rows if row["dataset"] == dataset_name]
        dataset_ttft = [row["ttft_seconds"] for row in dataset_rows]
        per_dataset[dataset_name] = {
            "count": len(dataset_rows),
            "mean_ttft_seconds": statistics.mean(dataset_ttft) if dataset_ttft else 0.0,
            "median_ttft_seconds": statistics.median(dataset_ttft) if dataset_ttft else 0.0,
            "p95_ttft_seconds": percentile(dataset_ttft, 95),
        }

    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in measured_rows:
        bucket_start = (row["prompt_tokens"] // bucket_width) * bucket_width
        bucket_end = bucket_start + bucket_width - 1
        label = f"{bucket_start}-{bucket_end}"
        buckets.setdefault(label, []).append(row)

    token_buckets = []
    for label in sorted(
        buckets.keys(),
        key=lambda item: int(item.split("-", 1)[0]),
    ):
        bucket_rows = buckets[label]
        bucket_ttft = [row["ttft_seconds"] for row in bucket_rows]
        token_buckets.append(
            {
                "token_range": label,
                "count": len(bucket_rows),
                "mean_ttft_seconds": statistics.mean(bucket_ttft),
                "median_ttft_seconds": statistics.median(bucket_ttft),
            }
        )

    return {
        "warmup_count": len(rows) - len(measured_rows),
        "measured_count": len(measured_rows),
        "mean_ttft_seconds": statistics.mean(ttft_values) if ttft_values else 0.0,
        "median_ttft_seconds": statistics.median(ttft_values) if ttft_values else 0.0,
        "p95_ttft_seconds": percentile(ttft_values, 95),
        "token_ttft_correlation": pearson_correlation(prompt_tokens, ttft_values),
        "token_bucket_width": bucket_width,
        "per_dataset": per_dataset,
        "token_buckets": token_buckets,
    }


def milliseconds(value: float) -> str:
    return f"{value * 1000:.2f}"


def format_correlation(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_summary_markdown(summary: dict[str, Any]) -> str:
    rag = summary["models"]["rag"]["summary"]
    block = summary["models"]["block"]["summary"]

    lines = [
        "# TTFT Summary",
        "",
        f"- Shared attention: `{summary['shared_attn_implementation']}`",
        "- Generation mode: deterministic `max_new_tokens=1`",
        f"- Median TTFT ratio (RAG / Block-FT): `{summary['median_ttft_ratio_rag_over_block']:.4f}`",
        f"- Mean TTFT ratio (RAG / Block-FT): `{summary['mean_ttft_ratio_rag_over_block']:.4f}`",
        "",
        "## Overall",
        "| Model | Warmups | Measured | Mean TTFT (ms) | Median TTFT (ms) | P95 TTFT (ms) | Token/TTFT Corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Tulu3-RAG | {rag['warmup_count']} | {rag['measured_count']} | "
            f"{milliseconds(rag['mean_ttft_seconds'])} | {milliseconds(rag['median_ttft_seconds'])} | "
            f"{milliseconds(rag['p95_ttft_seconds'])} | "
            f"{format_correlation(rag['token_ttft_correlation'])} |"
        ),
        (
            f"| Tulu3-Block-FT | {block['warmup_count']} | {block['measured_count']} | "
            f"{milliseconds(block['mean_ttft_seconds'])} | {milliseconds(block['median_ttft_seconds'])} | "
            f"{milliseconds(block['p95_ttft_seconds'])} | "
            f"{format_correlation(block['token_ttft_correlation'])} |"
        ),
        "",
        "## Per Dataset",
        "| Model | Dataset | Count | Mean TTFT (ms) | Median TTFT (ms) | P95 TTFT (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for model_label, model_summary in [
        ("Tulu3-RAG", rag),
        ("Tulu3-Block-FT", block),
    ]:
        for dataset_name in DATASET_NAMES:
            dataset_summary = model_summary["per_dataset"][dataset_name]
            lines.append(
                f"| {model_label} | {dataset_name} | {dataset_summary['count']} | "
                f"{milliseconds(dataset_summary['mean_ttft_seconds'])} | "
                f"{milliseconds(dataset_summary['median_ttft_seconds'])} | "
                f"{milliseconds(dataset_summary['p95_ttft_seconds'])} |"
            )

    lines.extend(
        [
            "",
            "## Prompt-Length Buckets",
            "| Model | Prompt Tokens | Count | Median TTFT (ms) | Mean TTFT (ms) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    for model_label, model_summary in [
        ("Tulu3-RAG", rag),
        ("Tulu3-Block-FT", block),
    ]:
        for bucket in model_summary["token_buckets"]:
            lines.append(
                f"| {model_label} | {bucket['token_range']} | {bucket['count']} | "
                f"{milliseconds(bucket['median_ttft_seconds'])} | "
                f"{milliseconds(bucket['mean_ttft_seconds'])} |"
            )

    return "\n".join(lines) + "\n"


def build_summary(
    *,
    subset_manifest_path: Path,
    rag_model_source: str,
    block_model_source: str,
    rag_per_example_path: Path,
    block_per_example_path: Path,
    shared_attn_implementation: str,
) -> dict[str, Any]:
    rag_rows = load_jsonl(rag_per_example_path)
    block_rows = load_jsonl(block_per_example_path)
    rag_summary = summarize_rows(rag_rows)
    block_summary = summarize_rows(block_rows)

    return {
        "subset_manifest": str(subset_manifest_path),
        "shared_attn_implementation": shared_attn_implementation,
        "models": {
            "rag": {
                "source": rag_model_source,
                "per_example_path": str(rag_per_example_path),
                "summary": rag_summary,
            },
            "block": {
                "source": block_model_source,
                "per_example_path": str(block_per_example_path),
                "summary": block_summary,
            },
        },
        "median_ttft_ratio_rag_over_block": (
            rag_summary["median_ttft_seconds"] / block_summary["median_ttft_seconds"]
            if block_summary["median_ttft_seconds"] > 0
            else 0.0
        ),
        "mean_ttft_ratio_rag_over_block": (
            rag_summary["mean_ttft_seconds"] / block_summary["mean_ttft_seconds"]
            if block_summary["mean_ttft_seconds"] > 0
            else 0.0
        ),
    }


def load_model_for_benchmark(
    *,
    model_path: str,
    gpu_id: int,
    attn_implementation: str,
):
    device = f"cuda:{gpu_id}"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=resolve_dtype(),
        device_map=device,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model


def run_internal_model_benchmark(args: argparse.Namespace) -> None:
    if not args.model_kind:
        raise ValueError("--model-kind is required in internal run-model mode")
    if not args.model_path or not args.subset_manifest or not args.per_example_output:
        raise ValueError("Internal run-model mode requires model path, subset manifest, and output path")

    subset_manifest = json.loads(Path(args.subset_manifest).read_text(encoding="utf-8"))
    data_root = Path(subset_manifest["data_root"]).resolve()
    model_path = str(Path(args.model_path).resolve())
    per_example_output_path = Path(args.per_example_output).resolve()
    if per_example_output_path.exists():
        per_example_output_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = load_model_for_benchmark(
        model_path=model_path,
        gpu_id=args.gpu_id,
        attn_implementation=args.attn_implementation,
    )
    generation_config = build_default_generation_config(
        tokenizer=tokenizer,
        max_new_tokens=1,
    )
    emb = None
    if args.model_kind == "block":
        emb = build_rotary_embedding(model_name_or_path=model_path, device=model.device)

    dataset_cache = {
        dataset_name: load_jsonl(dataset_path(data_root, dataset_name))
        for dataset_name in DATASET_NAMES
    }

    try:
        for phase_name in ["warmup", "measured"]:
            for dataset_name in DATASET_NAMES:
                for entry in subset_manifest["datasets"][dataset_name][phase_name]:
                    example = dataset_cache[dataset_name][entry["example_index"]]
                    documents = example["documents"]
                    question = example["question"]

                    if args.model_kind == "rag":
                        prompt = build_rag_prompt(question=question, documents=documents)
                        encoded = tokenizer(
                            prompt,
                            add_special_tokens=False,
                            return_tensors="pt",
                        )
                        input_ids = encoded["input_ids"].to(model.device)
                        attention_mask = encoded["attention_mask"].to(model.device)
                        input_length = input_ids.size(-1)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device=model.device)
                        start_time = time.perf_counter()
                        with torch.no_grad():
                            outputs = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                generation_config=generation_config,
                                use_cache=True,
                                tokenizer=tokenizer,
                            )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device=model.device)
                        ttft_seconds = time.perf_counter() - start_time
                        generated_token_ids = outputs[0][input_length:].detach().cpu()
                        prompt_tokens = entry["rag_prompt_tokens"]
                    else:
                        block_parts = build_rag_blocks(question=question, documents=documents)
                        encoded_inputs = encode_block_inputs(
                            blocks=block_parts[:-1],
                            instruction=block_parts[-1],
                            tokenizer=tokenizer,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device=model.device)
                        start_time = time.perf_counter()
                        generated_token_ids, _ = generate_block_tokens(
                            encoded_inputs=encoded_inputs,
                            generation_config=generation_config,
                            model=model,
                            emb=emb,
                            tokenizer=tokenizer,
                            num_local_attention_blocks=entry["num_local_attention_blocks"],
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device=model.device)
                        ttft_seconds = time.perf_counter() - start_time
                        prompt_tokens = entry["block_prompt_tokens"]

                    first_token_id = None
                    first_token_text = ""
                    if generated_token_ids.numel() > 0:
                        first_token_id = int(generated_token_ids[0].item())
                        first_token_text = decode_generated_tokens(
                            tokenizer=tokenizer,
                            token_ids=generated_token_ids[:1],
                        )

                    append_jsonl(
                        per_example_output_path,
                        {
                            "phase": phase_name,
                            "model_kind": args.model_kind,
                            "dataset": dataset_name,
                            "example_index": entry["example_index"],
                            "document_count": entry["document_count"],
                            "prompt_tokens": prompt_tokens,
                            "num_local_attention_blocks": entry["num_local_attention_blocks"],
                            "attn_implementation": args.attn_implementation,
                            "ttft_seconds": ttft_seconds,
                            "first_token_id": first_token_id,
                            "first_token_text": first_token_text,
                        },
                    )
    finally:
        del model
        gc_cuda()


def run_benchmark(args: argparse.Namespace) -> None:
    if args.samples_per_dataset < 1:
        raise ValueError("--samples-per-dataset must be at least 1")
    if args.warmup_per_dataset < 0:
        raise ValueError("--warmup-per-dataset must be non-negative")

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.prepare_data:
        run_prepare_data(data_root)

    rag_model_path = resolve_model_path(args.rag_model)
    block_model_path = resolve_model_path(args.block_model)
    shared_attn_implementation = resolve_shared_attention(
        rag_model_path=rag_model_path,
        block_model_path=block_model_path,
        gpu_id=args.gpu_id,
        requested_attn_implementation=args.attn_implementation,
    )

    subset_manifest_path = build_selection_manifest(
        rag_model_source=args.rag_model,
        block_model_source=args.block_model,
        rag_model_path=rag_model_path,
        block_model_path=block_model_path,
        data_root=data_root,
        output_root=output_root,
        seed=args.seed,
        warmup_per_dataset=args.warmup_per_dataset,
        samples_per_dataset=args.samples_per_dataset,
        attn_implementation=shared_attn_implementation,
    )

    rag_per_example_path = output_root / "rag_per_example.jsonl"
    block_per_example_path = output_root / "block_per_example.jsonl"
    run_model_subprocess(
        model_kind="rag",
        model_path=rag_model_path,
        subset_manifest_path=subset_manifest_path,
        per_example_output_path=rag_per_example_path,
        gpu_id=args.gpu_id,
        attn_implementation=shared_attn_implementation,
    )
    run_model_subprocess(
        model_kind="block",
        model_path=block_model_path,
        subset_manifest_path=subset_manifest_path,
        per_example_output_path=block_per_example_path,
        gpu_id=args.gpu_id,
        attn_implementation=shared_attn_implementation,
    )

    summary = build_summary(
        subset_manifest_path=subset_manifest_path,
        rag_model_source=args.rag_model,
        block_model_source=args.block_model,
        rag_per_example_path=rag_per_example_path,
        block_per_example_path=block_per_example_path,
        shared_attn_implementation=shared_attn_implementation,
    )
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.internal_mode == "probe-attention":
        if not args.model_path:
            raise ValueError("--model-path is required in internal probe-attention mode")
        probe_attention_support(
            model_path=str(Path(args.model_path).resolve()),
            gpu_id=args.gpu_id,
            attn_implementation=args.attn_implementation,
        )
        return
    if args.internal_mode == "run-model":
        run_internal_model_benchmark(args)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
