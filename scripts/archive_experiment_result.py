import argparse
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_RESULTS_DIR = ROOT_DIR / "experiment_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Run directory containing results.json and results.md")
    parser.add_argument("--name", default=None, help="Destination folder name under experiment_results")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source run directory not found: {source_dir}")

    source_json = source_dir / "results.json"
    source_md = source_dir / "results.md"
    if not source_json.is_file() or not source_md.is_file():
        raise FileNotFoundError(
            f"Expected results.json and results.md under {source_dir}"
        )

    destination_name = args.name or source_dir.name
    destination_dir = (EXPERIMENT_RESULTS_DIR / destination_name).resolve()
    destination_dir.parent.mkdir(parents=True, exist_ok=True)

    if destination_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Destination already exists: {destination_dir}. Use --overwrite to replace it."
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_json, destination_dir / "results.json")
    shutil.copy2(source_md, destination_dir / "results.md")

    print(f"Archived results to: {destination_dir}")


if __name__ == "__main__":
    main()
