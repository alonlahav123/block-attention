import argparse
import copy
import gc
import json
import random
import statistics
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
    build_block_past_key_values,
    build_rotary_embedding,
    count_block_prompt_tokens,
    decode_generated_tokens,
    encode_block_inputs,
    merge_and_rotary_past_key_values,
)
from src.rag_prompting import build_rag_blocks, build_rag_prompt

SAFE_WORDS = [
    "alpha",
    "amber",
    "anchor",
    "apex",
    "arc",
    "atlas",
    "aurora",
    "axis",
    "beacon",
    "birch",
    "bloom",
    "canyon",
    "cedar",
    "cipher",
    "clover",
    "comet",
    "coral",
    "delta",
    "dune",
    "echo",
    "ember",
    "falcon",
    "field",
    "fjord",
    "forest",
    "galaxy",
    "glacier",
    "grove",
    "harbor",
    "horizon",
    "jade",
    "jungle",
    "lagoon",
    "lantern",
    "lotus",
    "lumen",
    "maple",
    "meadow",
    "meteor",
    "monsoon",
    "nebula",
    "nectar",
    "onyx",
    "orchard",
    "origin",
    "pebble",
    "pine",
    "prairie",
    "quartz",
    "raven",
    "reef",
    "ridge",
    "river",
    "saffron",
    "sage",
    "sierra",
    "solstice",
    "sparrow",
    "summit",
    "tempest",
    "thicket",
    "timber",
    "topaz",
    "torrent",
    "valley",
    "velvet",
    "vertex",
    "violet",
    "willow",
    "zephyr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-model", default="ldsjmdy/Tulu3-RAG")
    parser.add_argument("--block-model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument(
        "--output-root",
        default=str(ROOT_DIR / "outputs" / "synthetic_ttft_32k"),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--target-prompt-tokens", type=int, default=32000)
    parser.add_argument("--num-documents", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "auto"],
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def gc_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda(device: torch.device | str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int(round((len(ordered) - 1) * (pct / 100.0))))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_seconds": statistics.mean(values) if values else 0.0,
        "median_seconds": statistics.median(values) if values else 0.0,
        "p95_seconds": percentile(values, 95),
    }


def milliseconds(value: float) -> str:
    return f"{value * 1000:.2f}"


def past_cache_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    for layer in past_key_values.layers:
        if layer.keys is not None:
            return int(layer.keys.shape[-2])
    return 0


@torch.no_grad()
def predict_first_token_ids(
    *,
    model,
    input_ids: torch.Tensor,
    past_key_values=None,
) -> torch.Tensor:
    total_length = past_cache_length(past_key_values) + input_ids.shape[1]
    attention_mask = torch.ones(
        (input_ids.shape[0], total_length),
        dtype=torch.int64,
        device=model.device,
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=False,
        return_dict=True,
    )
    return outputs.logits[:, -1, :].argmax(dim=-1).detach().cpu()


def random_chunk(rng: random.Random, words_per_chunk: int) -> str:
    words = [rng.choice(SAFE_WORDS) for _ in range(words_per_chunk)]
    return " ".join(words) + "."


def random_title(rng: random.Random, doc_index: int) -> str:
    return f"Synthetic Document {doc_index + 1}: {rng.choice(SAFE_WORDS).title()} {rng.choice(SAFE_WORDS).title()}"


def build_synthetic_documents(
    *,
    tokenizer,
    target_prompt_tokens: int,
    num_documents: int,
    seed: int,
) -> tuple[str, list[dict[str, Any]], int]:
    rng = random.Random(seed)
    question = "Return a one-word answer based on the retrieved documents."
    documents = [
        {"title": random_title(rng, doc_index), "text": ""}
        for doc_index in range(num_documents)
    ]

    chunk_words = 32
    round_robin_index = 0
    prompt_tokens = len(
        tokenizer.encode(
            build_rag_prompt(question=question, documents=documents),
            add_special_tokens=False,
        )
    )

    while prompt_tokens < target_prompt_tokens:
        document = documents[round_robin_index % num_documents]
        addition = random_chunk(rng, chunk_words)
        if document["text"]:
            document["text"] = f"{document['text']} {addition}"
        else:
            document["text"] = addition
        round_robin_index += 1
        prompt_tokens = len(
            tokenizer.encode(
                build_rag_prompt(question=question, documents=documents),
                add_special_tokens=False,
            )
        )

    return question, documents, prompt_tokens


def load_model_for_benchmark(
    *,
    model_path: Path,
    gpu_id: int,
    attn_implementation: str,
):
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(model_path),
        dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}",
        attn_implementation=attn_implementation,
    )


def can_load_model(
    *,
    model_path: Path,
    gpu_id: int,
    attn_implementation: str,
) -> bool:
    try:
        model = load_model_for_benchmark(
            model_path=model_path,
            gpu_id=gpu_id,
            attn_implementation=attn_implementation,
        )
        model.eval()
        del model
        gc_cuda()
        return True
    except Exception as exc:
        print(
            f"Failed to load {model_path.name} with {attn_implementation}: {exc}",
            flush=True,
        )
        gc_cuda()
        return False


def resolve_shared_attention(
    *,
    rag_model_path: Path,
    block_model_path: Path,
    gpu_id: int,
    requested_attn_implementation: str,
) -> str:
    candidates = (
        ["flash_attention_2", "sdpa"]
        if requested_attn_implementation == "auto"
        else [requested_attn_implementation]
    )
    for candidate in candidates:
        if not can_load_model(
            model_path=rag_model_path,
            gpu_id=gpu_id,
            attn_implementation=candidate,
        ):
            continue
        if not can_load_model(
            model_path=block_model_path,
            gpu_id=gpu_id,
            attn_implementation=candidate,
        ):
            continue
        return candidate
    raise RuntimeError("Could not find one attention implementation that works for both models")


def benchmark_rag(
    *,
    model_path: Path,
    attn_implementation: str,
    gpu_id: int,
    prompt: str,
    tokenizer,
    warmup_iters: int,
    measure_iters: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        output_path.unlink()

    model = load_model_for_benchmark(
        model_path=model_path,
        gpu_id=gpu_id,
        attn_implementation=attn_implementation,
    )
    model.eval()
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)

    measured_times: list[float] = []
    try:
        for phase_name, iterations in [
            ("warmup", warmup_iters),
            ("measured", measure_iters),
        ]:
            for iteration_index in range(iterations):
                sync_cuda(model.device)
                start_time = time.perf_counter()
                generated_token_ids = predict_first_token_ids(
                    model=model,
                    input_ids=input_ids,
                )
                sync_cuda(model.device)
                ttft_seconds = time.perf_counter() - start_time
                first_token_text = ""
                if generated_token_ids.numel() > 0:
                    first_token_text = decode_generated_tokens(
                        tokenizer=tokenizer,
                        token_ids=generated_token_ids[:1],
                    )

                append_jsonl(
                    output_path,
                    {
                        "phase": phase_name,
                        "iteration": iteration_index,
                        "ttft_seconds": ttft_seconds,
                        "first_token_text": first_token_text,
                    },
                )
                if phase_name == "measured":
                    measured_times.append(ttft_seconds)
    finally:
        del model
        gc_cuda()

    return {
        "prompt_tokens": int(input_ids.size(-1)),
        "timing": summarize(measured_times),
    }


def benchmark_block_precached(
    *,
    model_path: Path,
    attn_implementation: str,
    gpu_id: int,
    blocks: list[str],
    tokenizer,
    warmup_iters: int,
    measure_iters: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        output_path.unlink()

    model = load_model_for_benchmark(
        model_path=model_path,
        gpu_id=gpu_id,
        attn_implementation=attn_implementation,
    )
    model.eval()
    emb = build_rotary_embedding(model_name_or_path=str(model_path), device=model.device)
    encoded_inputs = encode_block_inputs(
        blocks=blocks[:-1],
        instruction=blocks[-1],
        tokenizer=tokenizer,
    )
    num_local_attention_blocks = len(blocks) - 1
    prompt_tokens = count_block_prompt_tokens(encoded_inputs)
    instruction_input_ids = torch.tensor(
        [encoded_inputs.instruction_token_ids],
        dtype=torch.int64,
        device=model.device,
    )

    measured_times: list[float] = []
    precache_times: list[float] = []
    try:
        for phase_name, iterations in [
            ("warmup", warmup_iters),
            ("measured", measure_iters),
        ]:
            for iteration_index in range(iterations):
                sync_cuda(model.device)
                precache_start = time.perf_counter()
                precached_past_key_values, _ = build_block_past_key_values(
                    encoded_inputs=encoded_inputs,
                    model=model,
                    emb=emb,
                    num_local_attention_blocks=num_local_attention_blocks,
                )
                sync_cuda(model.device)
                precache_seconds = time.perf_counter() - precache_start

                sync_cuda(model.device)
                start_time = time.perf_counter()
                merged_cache = None
                if precached_past_key_values is not None:
                    merged_cache = merge_and_rotary_past_key_values(
                        pkvs=copy.deepcopy(precached_past_key_values),
                        emb=emb,
                    )
                generated_token_ids = predict_first_token_ids(
                    model=model,
                    input_ids=instruction_input_ids,
                    past_key_values=merged_cache,
                )
                sync_cuda(model.device)
                ttft_seconds = time.perf_counter() - start_time

                first_token_text = ""
                if generated_token_ids.numel() > 0:
                    first_token_text = decode_generated_tokens(
                        tokenizer=tokenizer,
                        token_ids=generated_token_ids[:1],
                    )

                append_jsonl(
                    output_path,
                    {
                        "phase": phase_name,
                        "iteration": iteration_index,
                        "precache_seconds": precache_seconds,
                        "ttft_seconds": ttft_seconds,
                        "first_token_text": first_token_text,
                    },
                )
                if phase_name == "measured":
                    precache_times.append(precache_seconds)
                    measured_times.append(ttft_seconds)
    finally:
        del model
        gc_cuda()

    return {
        "prompt_tokens": prompt_tokens,
        "precache_timing": summarize(precache_times),
        "timing": summarize(measured_times),
    }


def benchmark_block_cache_ready(
    *,
    model_path: Path,
    attn_implementation: str,
    gpu_id: int,
    blocks: list[str],
    tokenizer,
    warmup_iters: int,
    measure_iters: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        output_path.unlink()

    model = load_model_for_benchmark(
        model_path=model_path,
        gpu_id=gpu_id,
        attn_implementation=attn_implementation,
    )
    model.eval()
    emb = build_rotary_embedding(model_name_or_path=str(model_path), device=model.device)
    encoded_inputs = encode_block_inputs(
        blocks=blocks[:-1],
        instruction=blocks[-1],
        tokenizer=tokenizer,
    )
    num_local_attention_blocks = len(blocks) - 1
    prompt_tokens = count_block_prompt_tokens(encoded_inputs)
    merged_cache_build_times: list[float] = []
    measured_times: list[float] = []
    instruction_input_ids = torch.tensor(
        [encoded_inputs.instruction_token_ids],
        dtype=torch.int64,
    )

    try:
        for phase_name, iterations in [
            ("warmup", warmup_iters),
            ("measured", measure_iters),
        ]:
            for iteration_index in range(iterations):
                sync_cuda(model.device)
                cache_build_start = time.perf_counter()
                block_past_key_values, _ = build_block_past_key_values(
                    encoded_inputs=encoded_inputs,
                    model=model,
                    emb=emb,
                    num_local_attention_blocks=num_local_attention_blocks,
                )
                merged_cache = None
                if block_past_key_values is not None:
                    merged_cache = merge_and_rotary_past_key_values(
                        pkvs=block_past_key_values,
                        emb=emb,
                    )
                sync_cuda(model.device)
                cache_build_seconds = time.perf_counter() - cache_build_start

                sync_cuda(model.device)
                start_time = time.perf_counter()
                generated_token_ids = predict_first_token_ids(
                    model=model,
                    input_ids=instruction_input_ids,
                    past_key_values=merged_cache,
                )
                sync_cuda(model.device)
                ttft_seconds = time.perf_counter() - start_time

                first_token_text = ""
                if generated_token_ids.numel() > 0:
                    first_token_text = decode_generated_tokens(
                        tokenizer=tokenizer,
                        token_ids=generated_token_ids[:1],
                    )

                append_jsonl(
                    output_path,
                    {
                        "phase": phase_name,
                        "iteration": iteration_index,
                        "cache_build_seconds": cache_build_seconds,
                        "ttft_seconds": ttft_seconds,
                        "first_token_text": first_token_text,
                    },
                )
                if phase_name == "measured":
                    merged_cache_build_times.append(cache_build_seconds)
                    measured_times.append(ttft_seconds)
    finally:
        del model
        gc_cuda()

    return {
        "prompt_tokens": prompt_tokens,
        "cache_ready_build_timing": summarize(merged_cache_build_times),
        "timing": summarize(measured_times),
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    rag = summary["rag"]
    block = summary["block_precached"]
    block_cache_ready = summary["block_cache_ready"]
    return "\n".join(
        [
            "# Synthetic Long-Context TTFT Benchmark",
            "",
            f"- Shared attention: `{summary['shared_attn_implementation']}`",
            "- `Block-FT (precached)` excludes per-document cache build but still includes merge-and-rotate inside the timer.",
            "- `Block-FT (cache-ready)` excludes both per-document cache build and merged-cache preparation, and measures only the final instruction prefill plus first token.",
            f"- Target prompt tokens: `{summary['target_prompt_tokens']}`",
            f"- Actual RAG prompt tokens: `{summary['rag_prompt_tokens']}`",
            f"- Actual Block prompt tokens: `{summary['block_prompt_tokens']}`",
            f"- Documents: `{summary['num_documents']}`",
            "",
            "## Overall",
            "| Path | Measured Runs | Mean TTFT (ms) | Median TTFT (ms) | P95 TTFT (ms) |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| Tulu3-RAG | {rag['timing']['count']} | "
                f"{milliseconds(rag['timing']['mean_seconds'])} | "
                f"{milliseconds(rag['timing']['median_seconds'])} | "
                f"{milliseconds(rag['timing']['p95_seconds'])} |"
            ),
            (
                f"| Tulu3-Block-FT (precached) | {block['timing']['count']} | "
                f"{milliseconds(block['timing']['mean_seconds'])} | "
                f"{milliseconds(block['timing']['median_seconds'])} | "
                f"{milliseconds(block['timing']['p95_seconds'])} |"
            ),
            (
                f"| Tulu3-Block-FT (cache-ready) | {block_cache_ready['timing']['count']} | "
                f"{milliseconds(block_cache_ready['timing']['mean_seconds'])} | "
                f"{milliseconds(block_cache_ready['timing']['median_seconds'])} | "
                f"{milliseconds(block_cache_ready['timing']['p95_seconds'])} |"
            ),
            "",
            "## Speedup",
            f"- Median speedup (RAG / Block-FT precached): `{summary['median_speedup_rag_over_block_precached']:.4f}`",
            f"- Mean speedup (RAG / Block-FT precached): `{summary['mean_speedup_rag_over_block_precached']:.4f}`",
            f"- Median speedup (RAG / Block-FT cache-ready): `{summary['median_speedup_rag_over_block_cache_ready']:.4f}`",
            f"- Mean speedup (RAG / Block-FT cache-ready): `{summary['mean_speedup_rag_over_block_cache_ready']:.4f}`",
            "",
            "## Block Precache Reference",
            (
                f"- Mean per-document cache build not included in `Block-FT (precached)` TTFT: "
                f"`{milliseconds(block['precache_timing']['mean_seconds'])} ms`"
            ),
            (
                f"- Median per-document cache build not included in `Block-FT (precached)` TTFT: "
                f"`{milliseconds(block['precache_timing']['median_seconds'])} ms`"
            ),
            (
                f"- Mean merged-cache prep not included in `Block-FT (cache-ready)` TTFT: "
                f"`{milliseconds(block_cache_ready['cache_ready_build_timing']['mean_seconds'])} ms`"
            ),
            (
                f"- Median merged-cache prep not included in `Block-FT (cache-ready)` TTFT: "
                f"`{milliseconds(block_cache_ready['cache_ready_build_timing']['median_seconds'])} ms`"
            ),
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rag_model_path = resolve_model_path(args.rag_model)
    block_model_path = resolve_model_path(args.block_model)
    rag_tokenizer = AutoTokenizer.from_pretrained(str(rag_model_path), use_fast=False)
    block_tokenizer = AutoTokenizer.from_pretrained(str(block_model_path), use_fast=False)

    rag_context_limit = resolve_context_limit(AutoConfig.from_pretrained(str(rag_model_path)))
    block_context_limit = resolve_context_limit(AutoConfig.from_pretrained(str(block_model_path)))
    effective_target_prompt_tokens = min(
        args.target_prompt_tokens,
        rag_context_limit - 64,
        block_context_limit - 64,
    )
    if effective_target_prompt_tokens < 1024:
        raise ValueError("Effective target prompt length is too small after context-limit adjustment")

    shared_attn_implementation = resolve_shared_attention(
        rag_model_path=rag_model_path,
        block_model_path=block_model_path,
        gpu_id=args.gpu_id,
        requested_attn_implementation=args.attn_implementation,
    )

    question, documents, rag_prompt_tokens = build_synthetic_documents(
        tokenizer=rag_tokenizer,
        target_prompt_tokens=effective_target_prompt_tokens,
        num_documents=args.num_documents,
        seed=args.seed,
    )
    prompt = build_rag_prompt(question=question, documents=documents)
    blocks = build_rag_blocks(question=question, documents=documents)
    block_prompt_tokens = count_block_prompt_tokens(
        encode_block_inputs(
            blocks=blocks[:-1],
            instruction=blocks[-1],
            tokenizer=block_tokenizer,
        )
    )

    write_json(
        output_root / "synthetic_prompt.json",
        {
            "seed": args.seed,
            "target_prompt_tokens": args.target_prompt_tokens,
            "effective_target_prompt_tokens": effective_target_prompt_tokens,
            "num_documents": args.num_documents,
            "question": question,
            "documents": documents,
            "rag_prompt_tokens": rag_prompt_tokens,
            "block_prompt_tokens": block_prompt_tokens,
        },
    )

    rag_summary = benchmark_rag(
        model_path=rag_model_path,
        attn_implementation=shared_attn_implementation,
        gpu_id=args.gpu_id,
        prompt=prompt,
        tokenizer=rag_tokenizer,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        output_path=output_root / "rag_runs.jsonl",
    )
    block_summary = benchmark_block_precached(
        model_path=block_model_path,
        attn_implementation=shared_attn_implementation,
        gpu_id=args.gpu_id,
        blocks=blocks,
        tokenizer=block_tokenizer,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        output_path=output_root / "block_precached_runs.jsonl",
    )
    block_cache_ready_summary = benchmark_block_cache_ready(
        model_path=block_model_path,
        attn_implementation=shared_attn_implementation,
        gpu_id=args.gpu_id,
        blocks=blocks,
        tokenizer=block_tokenizer,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        output_path=output_root / "block_cache_ready_runs.jsonl",
    )

    summary = {
        "shared_attn_implementation": shared_attn_implementation,
        "target_prompt_tokens": args.target_prompt_tokens,
        "effective_target_prompt_tokens": effective_target_prompt_tokens,
        "rag_prompt_tokens": rag_summary["prompt_tokens"],
        "block_prompt_tokens": block_summary["prompt_tokens"],
        "num_documents": args.num_documents,
        "rag": rag_summary,
        "block_precached": block_summary,
        "block_cache_ready": block_cache_ready_summary,
        "median_speedup_rag_over_block_precached": (
            rag_summary["timing"]["median_seconds"] / block_summary["timing"]["median_seconds"]
            if block_summary["timing"]["median_seconds"] > 0
            else 0.0
        ),
        "mean_speedup_rag_over_block_precached": (
            rag_summary["timing"]["mean_seconds"] / block_summary["timing"]["mean_seconds"]
            if block_summary["timing"]["mean_seconds"] > 0
            else 0.0
        ),
        "median_speedup_rag_over_block_cache_ready": (
            rag_summary["timing"]["median_seconds"] / block_cache_ready_summary["timing"]["median_seconds"]
            if block_cache_ready_summary["timing"]["median_seconds"] > 0
            else 0.0
        ),
        "mean_speedup_rag_over_block_cache_ready": (
            rag_summary["timing"]["mean_seconds"] / block_cache_ready_summary["timing"]["mean_seconds"]
            if block_cache_ready_summary["timing"]["mean_seconds"] > 0
            else 0.0
        ),
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )

    print(f"Shared attention: {shared_attn_implementation}")
    print(f"Wrote prompt manifest: {output_root / 'synthetic_prompt.json'}")
    print(f"Wrote summary: {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
