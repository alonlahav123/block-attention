from typing import List, TypedDict

Document = TypedDict("Document", {"title": str, "text": str, "score": float})

RAG_SYSTEM_PROMPT = (
    "You are an intelligent AI assistant. Please answer questions based on the user's "
    "instructions. Below are some reference documents that may help you in answering "
    "the user's question."
)

RAG_USER_PROMPT_TEMPLATE = (
    "Please write a high-quality answer for the given question using only the provided "
    "search documents (some of which might be irrelevant).\n"
    "Question: {question}"
)


def build_rag_user_prompt(question: str) -> str:
    return RAG_USER_PROMPT_TEMPLATE.format(question=question.strip())


def build_rag_document_blocks(documents: List[Document]) -> List[str]:
    blocks: List[str] = []
    for index, document in enumerate(documents):
        suffix = "\n" if index < len(documents) - 1 else ""
        blocks.append(f"- Title: {document['title']}\n{document['text'].strip()}{suffix}")
    return blocks


def build_rag_blocks(question: str, documents: List[Document]) -> List[str]:
    return [
        f"<|user|>\n{RAG_SYSTEM_PROMPT}\n\n",
        *build_rag_document_blocks(documents=documents),
        f"\n\n{build_rag_user_prompt(question=question)}\n<|assistant|>\n",
    ]


def build_rag_prompt(question: str, documents: List[Document]) -> str:
    system_content = (
        f"{RAG_SYSTEM_PROMPT}\n\n" + "".join(build_rag_document_blocks(documents=documents))
    ).strip()
    user_content = build_rag_user_prompt(question=question)
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}"
        "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
