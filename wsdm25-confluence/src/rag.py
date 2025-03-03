from typing import List

import numpy as np
from attribution import Attribution, AttributionOutput
from preprocessing.embedding import Document, embed_text, vector_search
from pydantic import BaseModel
from utils import chat_with_gpt


class RagResponse(BaseModel):
    query: str
    answer: str
    retrieved_docs: List[Document]
    attribution: List[str]


def rag(
    query: str,
    attributer: Attribution,
    documents: List[Document],
    top_k: int = 5,
    completion_model: str = "gpt-4o",
) -> RagResponse:

    embeddings = np.array(
        [doc.embedding for doc in documents if doc.embedding is not None]
    )

    query_vector = embed_text(query)
    top_indices = vector_search(query_vector, embeddings, top_k=top_k)
    top_docs = [documents[i] for i in top_indices]

    answer = chat_with_gpt(query, top_docs, completion_model)

    softmax_output, attribution_result, doc_mappings = attributer.get_attributions(
        query, top_docs, history=[], answer=answer.content
    )  # TODO: add history when available

    return RagResponse(
        query=query,
        answer=answer.content,
        retrieved_docs=top_docs,
        attribution=softmax_output,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(rag, as_positional=False)
