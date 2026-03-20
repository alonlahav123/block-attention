import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_synthetic_ttft import (  # noqa: E402
    milliseconds,
    resolve_model_path,
    resolve_shared_attention,
    run_synthetic_benchmark,
)

ROOT_DIR = SCRIPT_DIR.parent

PAPER_TABLE3 = [
    {"label": "50", "ttft_vanilla_ms": 26, "ttft_block_ms": 26},
    {"label": "512", "ttft_vanilla_ms": 50, "ttft_block_ms": 26},
    {"label": "1K", "ttft_vanilla_ms": 87, "ttft_block_ms": 26},
    {"label": "2K", "ttft_vanilla_ms": 167, "ttft_block_ms": 26},
    {"label": "4K", "ttft_vanilla_ms": 330, "ttft_block_ms": 27},
    {"label": "8K", "ttft_vanilla_ms": 691, "ttft_block_ms": 29},
    {"label": "16K", "ttft_vanilla_ms": 1515, "ttft_block_ms": 34},
    {"label": "32K", "ttft_vanilla_ms": 3638, "ttft_block_ms": 45},
]

DEFAULT_SWEEP = [
    ("50", 50),
    ("512", 512),
    ("1K", 1024),
    ("2K", 2048),
    ("4K", 4096),
    ("8K", 8192),
    ("16K", 16384),
    ("32K", 32000),
]
DEFAULT_SWEEP_CSV = ",".join(str(target_tokens) for _, target_tokens in DEFAULT_SWEEP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-model", default="ldsjmdy/Tulu3-RAG")
    parser.add_argument("--block-model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument(
        "--output-root",
        default=str(ROOT_DIR / "outputs" / "synthetic_ttft_table3_sweep"),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--user-input-tokens", type=int, default=50)
    parser.add_argument("--num-documents", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--passage-token-sweep",
        default=DEFAULT_SWEEP_CSV,
        help="Comma-separated passage token targets. The default mirrors Table 3 labels, with 32K approximated as 32000 tokens.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "auto"],
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_passage_token_sweep(raw_value: str) -> list[tuple[str, int]]:
    values: list[tuple[str, int]] = []
    for raw_piece in raw_value.split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        target = int(piece)
        label = format_length_label(target)
        values.append((label, target))
    if not values:
        raise ValueError("Expected at least one passage length in --passage-token-sweep")
    return values


def format_length_label(target_passage_tokens: int) -> str:
    if target_passage_tokens == 32000:
        return "32K"
    if target_passage_tokens >= 1024 and target_passage_tokens % 1024 == 0:
        return f"{target_passage_tokens // 1024}K"
    return str(target_passage_tokens)


def lookup_paper_row(label: str) -> dict[str, Any] | None:
    for row in PAPER_TABLE3:
        if row["label"] == label:
            return row
    return None


def format_median_ms(run_summary: dict[str, Any], path_key: str) -> str:
    seconds = run_summary[path_key]["timing"]["median_seconds"]
    return milliseconds(seconds)


def format_speedup(numerator_seconds: float, denominator_seconds: float) -> str:
    if denominator_seconds <= 0:
        return "0.00x"
    return f"{numerator_seconds / denominator_seconds:.2f}x"


def render_table_row(label: str, values: list[str]) -> str:
    return f"| {label} | " + " | ".join(values) + " |"


def render_summary_markdown(summary: dict[str, Any]) -> str:
    column_labels = [row["label"] for row in summary["runs"]]
    paper_vanilla = []
    paper_block = []
    rag = []
    block_precached = []
    block_cache_ready = []
    speedup_precached = []
    speedup_cache_ready = []

    for row in summary["runs"]:
        paper_row = lookup_paper_row(row["label"])
        paper_vanilla.append(
            str(paper_row["ttft_vanilla_ms"]) if paper_row is not None else "-"
        )
        paper_block.append(
            str(paper_row["ttft_block_ms"]) if paper_row is not None else "-"
        )
        rag.append(format_median_ms(row["summary"], "rag"))
        block_precached.append(format_median_ms(row["summary"], "block_precached"))
        block_cache_ready.append(format_median_ms(row["summary"], "block_cache_ready"))
        speedup_precached.append(
            format_speedup(
                row["summary"]["rag"]["timing"]["median_seconds"],
                row["summary"]["block_precached"]["timing"]["median_seconds"],
            )
        )
        speedup_cache_ready.append(
            format_speedup(
                row["summary"]["rag"]["timing"]["median_seconds"],
                row["summary"]["block_cache_ready"]["timing"]["median_seconds"],
            )
        )

    detail_rows = []
    for row in summary["runs"]:
        detail_rows.append(
            "| {label} | {target_passage_tokens} | {actual_passage_tokens} | {rag_prompt_tokens} | {block_prompt_tokens} | {rag_ms} | {block_precached_ms} | {block_cache_ready_ms} |".format(
                label=row["label"],
                target_passage_tokens=row["target_passage_tokens"],
                actual_passage_tokens=row["summary"]["actual_passage_tokens"],
                rag_prompt_tokens=row["summary"]["rag_prompt_tokens"],
                block_prompt_tokens=row["summary"]["block_prompt_tokens"],
                rag_ms=format_median_ms(row["summary"], "rag"),
                block_precached_ms=format_median_ms(row["summary"], "block_precached"),
                block_cache_ready_ms=format_median_ms(row["summary"], "block_cache_ready"),
            )
        )

    return "\n".join(
        [
            "# Synthetic TTFT Sweep",
            "",
            "- This sweep follows the Table 3 setup style: fixed user input length with increasing retrieved passage length.",
            f"- Shared attention: `{summary['shared_attn_implementation']}`",
            f"- User input tokens: `{summary['user_input_tokens']}`",
            f"- Documents: `{summary['num_documents']}`",
            f"- Warmups per setting: `{summary['warmup_iters']}`",
            f"- Measured runs per setting: `{summary['measure_iters']}`",
            "- Reported TTFT values below use the median across measured runs.",
            "- The local `32K` column uses a target passage length of `32000` tokens so the full prompt stays near the paper's 32K regime after prompt wrappers.",
            "",
            "## Table 3 Style Comparison",
            "| Path | " + " | ".join(column_labels) + " |",
            "| --- | " + " | ".join(["---:"] * len(column_labels)) + " |",
            render_table_row("Paper TTFT-vanilla (ms)", paper_vanilla),
            render_table_row("Paper TTFT-block (ms)", paper_block),
            render_table_row("Reproduction Tulu3-RAG (ms)", rag),
            render_table_row("Reproduction Tulu3-Block-FT precached (ms)", block_precached),
            render_table_row("Reproduction Tulu3-Block-FT cache-ready (ms)", block_cache_ready),
            render_table_row("Reproduction speedup RAG / precached", speedup_precached),
            render_table_row("Reproduction speedup RAG / cache-ready", speedup_cache_ready),
            "",
            "## Sweep Details",
            "| Label | Target Passage Tokens | Actual Passage Tokens | RAG Prompt Tokens | Block Prompt Tokens | RAG Median TTFT (ms) | Block Precached Median TTFT (ms) | Block Cache-Ready Median TTFT (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *detail_rows,
            "",
            "## Notes",
            "- Paper values come from Table 3 of the Block-Attention paper.",
            "- `precached` excludes per-document KV construction but still includes online merge-and-rotate.",
            "- `cache-ready` excludes both per-document KV construction and merged-cache preparation, so it is the tighter steady-state upper bound.",
            "- Each sweep point also writes a full per-setting summary under its own subdirectory.",
        ]
    ) + "\n"


def build_summary_payload(
    *,
    rag_model: str,
    block_model: str,
    output_root: Path,
    shared_attn_implementation: str,
    user_input_tokens: int,
    num_documents: int,
    warmup_iters: int,
    measure_iters: int,
    seed: int,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "rag_model": rag_model,
        "block_model": block_model,
        "output_root": str(output_root),
        "shared_attn_implementation": shared_attn_implementation,
        "user_input_tokens": user_input_tokens,
        "num_documents": num_documents,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "seed": seed,
        "paper_table3": PAPER_TABLE3,
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sweep = parse_passage_token_sweep(args.passage_token_sweep)
    rag_model_path = resolve_model_path(args.rag_model)
    block_model_path = resolve_model_path(args.block_model)
    shared_attn_implementation = resolve_shared_attention(
        rag_model_path=rag_model_path,
        block_model_path=block_model_path,
        gpu_id=args.gpu_id,
        requested_attn_implementation=args.attn_implementation,
    )

    runs: list[dict[str, Any]] = []
    for label, target_passage_tokens in sweep:
        case_output_root = output_root / f"passage_{target_passage_tokens}"
        case_summary = run_synthetic_benchmark(
            rag_model=args.rag_model,
            block_model=args.block_model,
            output_root=case_output_root,
            gpu_id=args.gpu_id,
            target_prompt_tokens=None,
            target_passage_tokens=target_passage_tokens,
            user_input_tokens=args.user_input_tokens,
            num_documents=args.num_documents,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            seed=args.seed,
            attn_implementation=args.attn_implementation,
            shared_attn_implementation=shared_attn_implementation,
        )
        runs.append(
            {
                "label": label,
                "target_passage_tokens": target_passage_tokens,
                "output_dir": str(case_output_root),
                "summary": case_summary,
            }
        )

    summary = build_summary_payload(
        rag_model=args.rag_model,
        block_model=args.block_model,
        output_root=output_root,
        shared_attn_implementation=shared_attn_implementation,
        user_input_tokens=args.user_input_tokens,
        num_documents=args.num_documents,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        seed=args.seed,
        runs=runs,
    )
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )

    print(f"Shared attention: {shared_attn_implementation}")
    print(f"Wrote summary: {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
