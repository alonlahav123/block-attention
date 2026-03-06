import os
import json

import fire
import regex
import string
import statistics

from tqdm import tqdm
from dataclasses import dataclass

from typing import Any, Dict, List, TypedDict

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


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(i) for i in f]


@dataclass
class EvalArgs:
    input: str



def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


METRICS = [(best_subspan_em, "best_subspan_em"),]



def get_metrics_for_example(example: SFTDataInstance):
    gold_answers = example["answers"]
    model_answer = example["generated"].split("<|im_end|>")[0].split("<|eot_id|>")[0]

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return example_metrics, example


def load_examples(path: str) -> List[SFTDataInstance]:
    if os.path.isfile(path):
        return load_jsonline(fp=path)

    all_examples: List[SFTDataInstance] = []
    for f_name in sorted(os.listdir(path)):
        fp = os.path.join(path, f_name)
        if os.path.isfile(fp):
            all_examples.extend(load_jsonline(fp=fp))
    return all_examples


def evaluate_examples(examples: List[SFTDataInstance]) -> Dict[str, float]:
    all_example_metrics = []
    for example in tqdm(examples, total=len(examples), desc="Eval: "):
        all_example_metrics.append(get_metrics_for_example(example=example))

    results: Dict[str, float] = {"count": float(len(examples))}
    for _, metric_name in METRICS:
        results[metric_name] = statistics.mean(metrics[metric_name] for metrics, _ in all_example_metrics)
    return results


def evaluate_path(path: str) -> Dict[str, float]:
    examples = load_examples(path=path)
    if not examples:
        raise ValueError(f"No examples found under: {path}")
    return evaluate_examples(examples=examples)


def main(args: EvalArgs):
    results = evaluate_path(path=args.input)
    print(f"count: {int(results['count'])}")
    for _, metric_name in METRICS:
        print(f"{metric_name}: {results[metric_name]:.6f}")



if __name__ == '__main__':
    args: EvalArgs = fire.Fire(component=EvalArgs)
    main(args=args)
