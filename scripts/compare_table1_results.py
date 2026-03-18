import argparse
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASETS = ["2wiki", "hqa", "nq", "tqa", "macro_average"]
PAPER_ROWS = {
    "Tulu3-Block-FT (paper)": {
        "2wiki": 0.722,
        "hqa": 0.723,
        "nq": 0.604,
        "tqa": 0.751,
        "macro_average": (0.722 + 0.723 + 0.604 + 0.751) / 4,
    },
    "Tulu3-RAG (paper)": {
        "2wiki": 0.732,
        "hqa": 0.748,
        "nq": 0.615,
        "tqa": 0.758,
        "macro_average": (0.732 + 0.748 + 0.615 + 0.758) / 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Result in the form label=path/to/results.json",
    )
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    return parser.parse_args()


def load_result(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "2wiki": payload["datasets"]["2wiki"]["best_subspan_em"],
        "hqa": payload["datasets"]["hqa"]["best_subspan_em"],
        "nq": payload["datasets"]["nq"]["best_subspan_em"],
        "tqa": payload["datasets"]["tqa"]["best_subspan_em"],
        "macro_average": payload["macro_average"],
    }


def parse_result_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected --result label=path, got: {value}")
    label, raw_path = value.split("=", 1)
    return label.strip(), Path(raw_path).resolve()


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}"


def render_absolute_table(rows: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "## Absolute Scores",
        "| Run | 2wiki | HQA | NQ | TQA | Macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, scores in rows.items():
        lines.append(
            f"| {label} | {format_pct(scores['2wiki'])} | {format_pct(scores['hqa'])} | "
            f"{format_pct(scores['nq'])} | {format_pct(scores['tqa'])} | {format_pct(scores['macro_average'])} |"
        )
    return lines


def paper_target_for_label(label: str) -> str | None:
    lower = label.lower()
    if "block" in lower:
        return "Tulu3-Block-FT (paper)"
    if "rag" in lower:
        return "Tulu3-RAG (paper)"
    return None


def render_delta_vs_paper(local_rows: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "## Delta Vs Paper",
        "| Run | Paper Target | 2wiki | HQA | NQ | TQA | Macro |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    added = False
    for label, scores in local_rows.items():
        target_label = paper_target_for_label(label)
        if target_label is None:
            continue
        paper_scores = PAPER_ROWS[target_label]
        lines.append(
            f"| {label} | {target_label} | "
            f"{(scores['2wiki'] - paper_scores['2wiki']) * 100:+.2f} | "
            f"{(scores['hqa'] - paper_scores['hqa']) * 100:+.2f} | "
            f"{(scores['nq'] - paper_scores['nq']) * 100:+.2f} | "
            f"{(scores['tqa'] - paper_scores['tqa']) * 100:+.2f} | "
            f"{(scores['macro_average'] - paper_scores['macro_average']) * 100:+.2f} |"
        )
        added = True
    return lines if added else []


def render_pairwise_delta(local_rows: dict[str, dict[str, float]]) -> list[str]:
    labels = list(local_rows.keys())
    if len(labels) != 2:
        return []
    left, right = labels
    left_scores = local_rows[left]
    right_scores = local_rows[right]
    lines = [
        f"## Delta: {right} - {left}",
        "| 2wiki | HQA | NQ | TQA | Macro |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {(right_scores['2wiki'] - left_scores['2wiki']) * 100:+.2f} | "
            f"{(right_scores['hqa'] - left_scores['hqa']) * 100:+.2f} | "
            f"{(right_scores['nq'] - left_scores['nq']) * 100:+.2f} | "
            f"{(right_scores['tqa'] - left_scores['tqa']) * 100:+.2f} | "
            f"{(right_scores['macro_average'] - left_scores['macro_average']) * 100:+.2f} |"
        ),
    ]
    return lines


def main() -> None:
    args = parse_args()
    if not args.result:
        raise ValueError("Provide at least one --result label=path/to/results.json")

    local_rows: dict[str, dict[str, float]] = {}
    for value in args.result:
        label, path = parse_result_arg(value)
        if not path.is_file():
            raise FileNotFoundError(f"Results file not found: {path}")
        local_rows[label] = load_result(path)

    all_rows = {**local_rows, **PAPER_ROWS}
    lines: list[str] = []
    lines.extend(render_absolute_table(all_rows))

    delta_vs_paper = render_delta_vs_paper(local_rows)
    if delta_vs_paper:
        lines.extend(["", *delta_vs_paper])

    pairwise_delta = render_pairwise_delta(local_rows)
    if pairwise_delta:
        lines.extend(["", *pairwise_delta])

    markdown = "\n".join(lines) + "\n"
    print(markdown)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote comparison to: {output_path}")


if __name__ == "__main__":
    main()
