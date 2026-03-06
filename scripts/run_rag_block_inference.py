import argparse
import json
import os

import requests

from tqdm import tqdm
from typing import Any, List, TypedDict

from src.rag_prompting import Document, build_rag_blocks


SFTDataInstanceInputs = TypedDict("SFTDataInstanceInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTDataInstance = TypedDict("SFTDataInstance", {
    "prompt": str,
    "question": str,
    "answers": List[str],
    "generated": str,
    "inputs": SFTDataInstanceInputs,
    "documents": List[Document]
})


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonline(fp: str, obj: List[Any]) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        for item in obj:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--server-url", required=True, type=str)
    parser.add_argument("--num-local-attention-blocks", default=10000, type=int)
    parser.add_argument("--timeout-seconds", default=600, type=int)
    return parser.parse_args()


def resolve_records(input_fp: str, output_fp: str) -> List[SFTDataInstance]:
    input_records: List[SFTDataInstance] = load_jsonline(fp=input_fp)
    if not os.path.exists(output_fp):
        return input_records

    output_records: List[SFTDataInstance] = load_jsonline(fp=output_fp)
    if len(input_records) != len(output_records):
        raise ValueError("Existing output file has a different number of records from the input file.")
    return output_records


def get_response(
    example: SFTDataInstance,
    server_url: str,
    num_local_attention_blocks: int,
    timeout_seconds: int,
) -> str:
    response = requests.post(
        url=server_url,
        json={
            "blocks": build_rag_blocks(question=example["question"], documents=example["documents"]),
            "num_local_attention_blocks": num_local_attention_blocks,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if "generated" not in payload:
        raise ValueError(f"Response payload is missing 'generated': {payload}")
    return payload["generated"]


def main() -> None:
    args = parse_args()
    records = resolve_records(input_fp=args.input, output_fp=args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for example in tqdm(records, total=len(records), desc="Generate RAG"):
        if example.get("generated", ""):
            continue
        example["generated"] = get_response(
            example=example,
            server_url=args.server_url,
            num_local_attention_blocks=args.num_local_attention_blocks,
            timeout_seconds=args.timeout_seconds,
        )
        write_jsonline(fp=args.output, obj=records)

    write_jsonline(fp=args.output, obj=records)


if __name__ == "__main__":
    main()
