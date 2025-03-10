import json
from pathlib import Path
from typing import List

from models.data import Document
from preprocessing.embedding import (
    batch_embed_documents,
    get_openai_api_key,
    save_documents,
)
from preprocessing.heterogenous_data.entrypoint import run_pipeline
from preprocessing.model import MultiModalConfig, VerbalizerDocument


class WebDocument(Document):
    url: str
    content: str
    date: str
    disclaimer: str
    space: str

def fetch_documents_from_folder(folder_path: Path) -> List[WebDocument]:
    documents = []
    for json_file in folder_path.glob("*.json"):
        with open(json_file, "r") as file:
            json_data = json.load(file)
        document = WebDocument(**json_data)
        documents.append(document)
    return documents


def prepare(
    multi_modal_config: MultiModalConfig,
    out_dir: Path = Path("out/confluence-openxt"),
    input_folder: Path = Path("confquestions/documents"),
):
    embedded_documents_file = Path(out_dir / "embedded_documents.npy")

    if not embedded_documents_file.exists():
        get_openai_api_key()

        documents = fetch_documents_from_folder(input_folder)
        verbalized_documents = run_pipeline(
            documents,
            db_path=None,
            out_dir=out_dir,
            verbalizer_config=multi_modal_config.verbalizer_config,
            modalities=multi_modal_config.modalities,
        )

        batch_embed_documents(verbalized_documents)
        save_documents(verbalized_documents, embedded_documents_file)

    else:
        print(
            f"Embedded documents file `{embedded_documents_file}` already exists. You can start chatting."
            f"To embedd new documents, please delete the file first or specify another out_dir."
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare, as_positional=False)
