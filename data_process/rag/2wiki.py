import re
import os
import json
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import random
import pandas as pd

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModel

from src.rag_prompting import build_rag_prompt
from src.rag_schema import normalize_answers
from src.runtime import get_cuda_device

Document = TypedDict("Document", {"title": str, "text": str, "score": float})

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


@dataclass
class BuildArgs:
    train_fp: Optional[str]
    eval_fp: Optional[str]
    output_dir: str


def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = token_embeddings.masked_fill_(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


@torch.no_grad()
def compute_embeddings(sentences: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(device=model.device, dtype=torch.int64)
    outputs = model(**inputs)
    embeddings = mean_pooling(token_embeddings=outputs[0], mask=inputs["attention_mask"])
    return embeddings


def process_instance(ins: Dict[str, Any]) -> SFTDataInstance:
    if isinstance(ins['context'], str):
        ins['context'] = json.loads(ins['context'])

    documents = [Document(title=i[0], text=''.join(i[1]), score=0.0) for i in ins["context"]]
    embeddings = compute_embeddings(
        sentences=[ins['question']] + [i['text'] for i in documents], model=model, tokenizer=retrieval_tokenizer
    )
    q_emb = embeddings[0].clone().unsqueeze(dim=0)
    scores = torch.matmul(input=q_emb, other=embeddings[1:].T).squeeze(dim=0)
    values, indices = torch.sort(input=scores, descending=True)
    values, indices = values.tolist(), indices.tolist()

    for idx, score in zip(indices, values):
        documents[idx]['score'] = score
    documents.sort(key=lambda i: i['score'], reverse=True)

    return SFTDataInstance(
        prompt="",
        question=ins['question'],
        answers=normalize_answers(ins),
        generated='',
        inputs=SFTDataInstanceInputs(input_ids=[], labels=[]),
        documents=documents[:10]
    )


def tokenizer_instance(ins: SFTDataInstance) -> SFTDataInstance:
    ins["prompt"] = build_rag_prompt(question=ins["question"], documents=ins["documents"])
    return ins


def process_file(input_file: str, output_file: str, num_samples: int):
    df = pd.read_parquet(path=input_file)
    wiki_instances: List[Dict[str, Any]] = df.to_dict(orient="records")
    if num_samples != -1:
        wiki_instances = random.sample(population=wiki_instances, k=num_samples)

    dataset: List[SFTDataInstance] = []
    for i in tqdm(range(0, len(wiki_instances)), desc="Process 2wiki: ", total=len(wiki_instances)):
        ins = process_instance(ins=wiki_instances[i])
        ins = tokenizer_instance(ins=ins)
        dataset.append(ins)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in dataset:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


def parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fp", type=str, default="")
    parser.add_argument("--eval_fp", "--dev_fp", dest="eval_fp", type=str, default="")
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    return BuildArgs(
        train_fp=args.train_fp or None, eval_fp=args.eval_fp or None, output_dir=args.output_dir
    )


if __name__ == '__main__':
    args = parse_args()
    os.system(f"mkdir -p {os.path.join(args.output_dir, '2wiki_train')}")
    os.system(f"mkdir -p {os.path.join(args.output_dir, '2wiki_eval')}")

    random.seed(42)
    model_name = "facebook/contriever-msmarco"
    retrieval_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model: PreTrainedModel = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        dtype=torch.bfloat16,
        device_map=get_cuda_device()
    )

    if args.train_fp is None and args.eval_fp is None:
        raise ValueError("At least one of --train_fp or --eval_fp must be provided.")

    if args.train_fp is not None:
        process_file(
            input_file=args.train_fp, output_file=os.path.join(args.output_dir, "2wiki_train", "dataset"), num_samples=-1
        )
    if args.eval_fp is not None:
        process_file(
            input_file=args.eval_fp, output_file=os.path.join(args.output_dir, "2wiki_eval", "dataset"), num_samples=-1
        )
