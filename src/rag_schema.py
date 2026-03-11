from typing import Any, Dict, List


def normalize_answers(instance: Dict[str, Any]) -> List[str]:
    answers = instance.get("answers")
    if isinstance(answers, list):
        normalized_answers = [str(answer) for answer in answers if str(answer)]
        if normalized_answers:
            return normalized_answers
    elif isinstance(answers, str) and answers:
        return [answers]

    answer = instance.get("answer")
    if isinstance(answer, list):
        normalized_answers = [str(item) for item in answer if str(item)]
        if normalized_answers:
            return normalized_answers
    elif answer is not None:
        answer_text = str(answer)
        if answer_text:
            return [answer_text]

    raise KeyError("Expected 'answer' or 'answers' in dataset instance.")
