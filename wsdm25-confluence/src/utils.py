from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from preprocessing.embedding import Document


def chat_with_gpt(
    query: str, context_docs: List[Document], model: str = "gpt-4"
) -> ChatCompletionMessage:
    client = OpenAI()

    prompt = """
            You are a helpful assistant. You are specialized in answering conversational questions in a retrieval-augmented generation (RAG) setup. Please provide as precise and concise an answer to the input question as possible (less than 50 words if possible), using the retrieved evidences in this prompt as sources for answering. There is no need to provide additional information beyond the requested answer, and also no need for supporting explanations. If the requested information cannot be found in the provided evidences, please state exactly: "The desired information cannot be found in the retrieved pool of evidence." Please use only the information presented in the evidences, and mark the sources used in your answering within square brackets, like [Source 2] or [Source 5]. Please do not use your parametric memory and world knowledge.
            """
    for doc in context_docs:
        prompt += f"- {doc.title}: {doc.content}\n\n"
    prompt += f"QUERY: {query}\n"

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return completion.choices[0].message


def display_retrieved_docs(rag_response, top_k=5):
    print(f"\n=== 🔍 Top {top_k} Retrieved Evidences ===\n")

    for i, document in enumerate(rag_response.retrieved_docs[:top_k], 1):
        doc_url = document.url
        doc_title = document.title
        doc_content = document.content

        print(f"({i}) {doc_title}")
        print(f"🔗 URL: {doc_url}")
        print(f"📄 Content Preview: {doc_content[:200]}{'...' if len(doc_content) > 200 else ''}")
        print("-" * 50)