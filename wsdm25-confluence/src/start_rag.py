from pathlib import Path
from attribution import Attribution
from preprocessing.embedding import get_openai_api_key, load_documents
from rag import rag
from utils import display_retrieved_docs


def main(
    completion_model: str = "gpt-4o",
    top_k: int = 10,
):

    get_openai_api_key()
    embedded_documents_file = Path("out/confluence-openxt/embedded_documents.npy")
    if not embedded_documents_file.exists() or not (documents := load_documents(embedded_documents_file)):
        raise ValueError("Embedded documents file is missing, or documents")

    print(f"{len(documents)} evidences have been loaded.")
    attributer = Attribution(completion_model)
    print(f"Attributer has been loaded with {attributer.model_name}")

    print("💬 Type your questions below.")
    print("🚪 Type 'exit' or 'quit' to leave the chatbot.")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Exiting the chatbot. Goodbye!")
            break

        rag_response = rag(
            query=question,
            attributer=attributer,
            documents=documents,
            top_k=top_k,
            completion_model=completion_model,
        )
        display_retrieved_docs(rag_response, top_k)
        print(f"\n💡 Answer: {rag_response.answer}\n")
        print(f"=== Attribution ===\n{rag_response.attribution}\n\n")


if __name__ == "__main__":
    main()
