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

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModel

from src.rag_prompting import build_rag_prompt
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
    eval_fp: str
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
    documents = [Document(title=i['title'], text=i['text'], score=0.0) for i in ins['ctxs']]
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
        answers=[ins['answer']],
        generated='',
        inputs=SFTDataInstanceInputs(input_ids=[], labels=[]),
        documents=documents[:10]
    )


def tokenizer_instance(ins: SFTDataInstance) -> SFTDataInstance:
    ins["prompt"] = build_rag_prompt(question=ins["question"], documents=ins["documents"])
    return ins


def process_file(input_file: str, output_file: str, num_samples: int):
    with open(input_file, "r", encoding="utf-8") as f:
        nq_instances: List[Dict[str, Any]] = json.load(f)
    if num_samples != -1:
        nq_instances = random.sample(population=nq_instances, k=num_samples)

    dataset: List[SFTDataInstance] = []
    for i in tqdm(range(0, len(nq_instances)), desc="Process NQ: ", total=len(nq_instances)):
        ins = process_instance(ins=nq_instances[i])
        ins = tokenizer_instance(ins=ins)
        dataset.append(ins)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in dataset:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


def parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_fp", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    return BuildArgs(eval_fp=args.eval_fp, output_dir=args.output_dir)


if __name__ == '__main__':
    args = parse_args()
    os.system(f"mkdir -p {os.path.join(args.output_dir, 'nq_eval')}")

    random.seed(42)
    model_name = "facebook/contriever-msmarco"
    retrieval_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model: PreTrainedModel = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        device_map=get_cuda_device()
    )

    process_file(
        input_file=args.eval_fp, output_file=os.path.join(args.output_dir, "nq_eval", "dataset"), num_samples=-1
    )
