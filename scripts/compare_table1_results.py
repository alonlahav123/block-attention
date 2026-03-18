import argparse
import json
from pathlib import Path

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
    parser.add_argument(
        "--stdout-format",
        choices=["plain", "markdown"],
        default="plain",
        help="Format printed to stdout. File output remains markdown.",
    )
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


def paper_target_for_label(label: str) -> str | None:
    lower = label.lower()
    if "block" in lower:
        return "Tulu3-Block-FT (paper)"
    if "rag" in lower:
        return "Tulu3-RAG (paper)"
    return None


def find_matching_row(
    rows: dict[str, dict[str, float]], keyword: str
) -> tuple[str, dict[str, float]] | None:
    for label, scores in rows.items():
        if keyword in label.lower():
            return label, scores
    return None


def make_absolute_section(rows: dict[str, dict[str, float]]) -> tuple[str, list[str], list[list[str]]]:
    section_rows: list[list[str]] = []
    for label, scores in rows.items():
        section_rows.append(
            [
                label,
                format_pct(scores["2wiki"]),
                format_pct(scores["hqa"]),
                format_pct(scores["nq"]),
                format_pct(scores["tqa"]),
                format_pct(scores["macro_average"]),
            ]
        )
    return "Absolute Scores", ["Run", "2wiki", "HQA", "NQ", "TQA", "Macro"], section_rows


def make_block_vs_rag_delta_section(
    local_rows: dict[str, dict[str, float]]
) -> tuple[str, list[str], list[list[str]]] | None:
    local_block = find_matching_row(local_rows, "block")
    local_rag = find_matching_row(local_rows, "rag")
    if local_block is None or local_rag is None:
        return None

    _, local_block_scores = local_block
    _, local_rag_scores = local_rag
    paper_block_scores = PAPER_ROWS["Tulu3-Block-FT (paper)"]
    paper_rag_scores = PAPER_ROWS["Tulu3-RAG (paper)"]

    return (
        "Block-FT - RAG Delta",
        ["Source", "2wiki", "HQA", "NQ", "TQA", "Macro"],
        [
            [
                "Your Runs",
                f"{(local_block_scores['2wiki'] - local_rag_scores['2wiki']) * 100:+.2f}",
                f"{(local_block_scores['hqa'] - local_rag_scores['hqa']) * 100:+.2f}",
                f"{(local_block_scores['nq'] - local_rag_scores['nq']) * 100:+.2f}",
                f"{(local_block_scores['tqa'] - local_rag_scores['tqa']) * 100:+.2f}",
                f"{(local_block_scores['macro_average'] - local_rag_scores['macro_average']) * 100:+.2f}",
            ],
            [
                "Paper",
                f"{(paper_block_scores['2wiki'] - paper_rag_scores['2wiki']) * 100:+.2f}",
                f"{(paper_block_scores['hqa'] - paper_rag_scores['hqa']) * 100:+.2f}",
                f"{(paper_block_scores['nq'] - paper_rag_scores['nq']) * 100:+.2f}",
                f"{(paper_block_scores['tqa'] - paper_rag_scores['tqa']) * 100:+.2f}",
                f"{(paper_block_scores['macro_average'] - paper_rag_scores['macro_average']) * 100:+.2f}",
            ],
        ],
    )


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    alignments = []
    for index, _ in enumerate(headers):
        alignments.append("---" if index == 0 else "---:")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_plain_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        cells = []
        for index, cell in enumerate(row):
            if index == 0:
                cells.append(cell.ljust(widths[index]))
            else:
                cells.append(cell.rjust(widths[index]))
        return "  ".join(cells)

    separator = "  ".join("-" * width for width in widths)
    return [format_row(headers), separator, *[format_row(row) for row in rows]]


def render_sections_markdown(
    sections: list[tuple[str, list[str], list[list[str]]]]
) -> str:
    lines: list[str] = []
    for index, (title, headers, rows) in enumerate(sections):
        if index:
            lines.append("")
        lines.append(f"## {title}")
        lines.extend(render_markdown_table(headers, rows))
    return "\n".join(lines) + "\n"


def render_sections_plain(
    sections: list[tuple[str, list[str], list[list[str]]]]
) -> str:
    lines: list[str] = []
    for index, (title, headers, rows) in enumerate(sections):
        if index:
            lines.append("")
        lines.append(title)
        lines.extend(render_plain_table(headers, rows))
    return "\n".join(lines) + "\n"


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
    sections: list[tuple[str, list[str], list[list[str]]]] = [
        make_absolute_section(all_rows)
    ]

    block_vs_rag_delta = make_block_vs_rag_delta_section(local_rows)
    if block_vs_rag_delta is not None:
        sections.append(block_vs_rag_delta)

    markdown = render_sections_markdown(sections)
    plain = render_sections_plain(sections)

    if args.stdout_format == "markdown":
        print(markdown, end="")
    else:
        print(plain, end="")

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote comparison to: {output_path}")


if __name__ == "__main__":
    main()
