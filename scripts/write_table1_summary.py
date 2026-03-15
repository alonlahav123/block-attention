import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag_eval import evaluate_path


DATASET_NAMES = ["2wiki", "hqa", "nq", "tqa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--2wiki", dest="two_wiki", required=True)
    parser.add_argument("--hqa", required=True)
    parser.add_argument("--nq", required=True)
    parser.add_argument("--tqa", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix).resolve()
    dataset_paths = {
        "2wiki": args.two_wiki,
        "hqa": args.hqa,
        "nq": args.nq,
        "tqa": args.tqa,
    }

    scores: dict[str, dict[str, float]] = {}
    for dataset_name in DATASET_NAMES:
        result = evaluate_path(dataset_paths[dataset_name])
        scores[dataset_name] = {
            "count": int(result["count"]),
            "best_subspan_em": result["best_subspan_em"],
        }

    macro_average = sum(item["best_subspan_em"] for item in scores.values()) / len(scores)
    summary = {
        "datasets": scores,
        "macro_average": macro_average,
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    markdown_lines = [
        "| Dataset | Count | best_subspan_em |",
        "| --- | ---: | ---: |",
    ]
    for dataset_name in DATASET_NAMES:
        item = scores[dataset_name]
        markdown_lines.append(
            f"| {dataset_name} | {item['count']} | {item['best_subspan_em']:.6f} |"
        )
    markdown_lines.append(f"| macro_average | - | {macro_average:.6f} |")
    output_prefix.with_suffix(".md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
