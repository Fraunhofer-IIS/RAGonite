import os
from pathlib import Path
from typing import List

import numpy as np
from models.data import Document
from openai import OpenAI


def get_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        os.environ["OPENAI_API_KEY"] = api_key


def batch_embed_documents(
    documents: List[Document],
    model: str = "text-embedding-3-small",
    max_tokens: int = 8192,  # defaults to text-embedding-3-small context
    batch_size: int = 16,
):
    client = OpenAI()
    print("Start embedding documents...")
    for i in range(0, len(documents), batch_size):
        print(f"Embedding batch {int(i//batch_size) + 1}")
        batch = documents[i : i + batch_size]

        contents = []
        for doc in batch:
            token_estimate = (
                len(doc.content) // 4
            )  # Approx. 4 chars per token. Please note this is a very rough estimation to avoid bloating this up.
            if token_estimate > max_tokens:
                print(f"Skipping document ID {doc.id}: exceeds {max_tokens} tokens.")
                continue
            contents.append(doc.content)

        if not contents:
            print("No documents in this batch fit within the token limit.")
            continue

        response = client.embeddings.create(model=model, input=contents)
        embeddings = [item.embedding for item in response.data]

        for doc, embedding in zip(batch, embeddings):
            doc.embedding = np.array(embedding)


def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
) -> np.ndarray:
    client = OpenAI()
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding)


def vector_search(
    query_embed: np.ndarray, embeddings: np.ndarray, top_k: int = 3
) -> List[int]:
    if query_embed is None or not isinstance(query_embed, np.ndarray):
        raise ValueError("query_embed must be a non-None numpy array.")

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("embeddings must be a non-empty numpy array.")

    if len(query_embed.shape) != 1:
        raise ValueError("query_embed must be a 1D array.")

    if len(embeddings.shape) != 2:
        raise ValueError("embeddings must be a 2D array.")

    query_norm = query_embed / np.linalg.norm(query_embed)
    docs_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(np.isnan(docs_norm)) or np.any(np.isnan(query_norm)):
        raise ValueError(
            "Normalization failed; embeddings may contain zero vectors or invalid values."
        )

    similarities = docs_norm @ query_norm
    return np.argsort(-similarities)[:top_k]


def save_documents(documents: List[Document], out_file: Path):
    for doc in documents:
        if doc.embedding is None:
            print(f"Document ID: {doc.id} has no embedding!")
    np.save(str(out_file), documents, allow_pickle=True)
    print(f"Documents saved to {out_file}")


def load_documents(file_path: Path) -> List[Document]:
    return np.load(str(file_path), allow_pickle=True).tolist()
