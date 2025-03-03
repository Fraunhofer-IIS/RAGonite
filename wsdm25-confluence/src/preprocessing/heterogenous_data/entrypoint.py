import json
from pathlib import Path
from typing import Optional, List
from preprocessing.heterogenous_data.extractor import parse_html_content, PassageExtractor
from models.data import Document

from preprocessing.utils import generate_heterogenous_processing_summary_from_passages

from preprocessing.model import Passage
from preprocessing.heterogenous_data.contextualization import assemble_passage_text


def run_pipeline(
    documents: List[Document],
    verbalizer_config,
    out_dir: Path,
    db_path: Optional[Path] = None,
    modalities: Optional[List] = None,
    debug_mode: bool = True
) -> List[Document]:
    """
    Processes documents, extracts passages, and optionally generates an HTML debugging report.
    Each passage becomes a separate document with a unique ID.
    """
    processed_documents = []
    passage_container = []
    html_content = _build_html_header() if debug_mode else []

    for document_idx, document in enumerate(documents):
        soup = parse_html_content(document.content)

        extractor = PassageExtractor(verbalizer_config, document, modalities, db_path)
        passages = extractor.extract_passages(soup)

        if document.space:
            for psg in passages:
                psg.space = document.space
        passage_container.extend(passages)

        for passage in passages:
            passage = assemble_passage_text(passage, verbalizer_config.context_mode)
            processed_documents.append(
                Document(
                    id=passage.passage_id,
                    title=passage.page_title,
                    content=passage.content,
                    url=document.url,
                    attachment=passage.attachment,
                )
            )

        if debug_mode:
            html_content.extend(generate_page_debugging_content(document, passages, verbalizer_config.context_mode))

    summary_text = generate_heterogenous_processing_summary_from_passages(passage_container, len(processed_documents))

    out_dir.mkdir(parents=True, exist_ok=True)
    documents_output_path = out_dir / "processed_documents.json"
    with documents_output_path.open("w", encoding="utf-8") as f:
        json.dump([doc.dict() for doc in processed_documents], f, ensure_ascii=False, indent=4)

    summary_output_path = out_dir / "summary.txt"
    summary_output_path.write_text(summary_text, encoding="utf-8")

    if debug_mode:
        html_content.append("</body>")
        html_content.append("</html>")
        debug_output_path = out_dir / "debug_output.html"
        debug_output_path.write_text("\n".join(html_content), encoding="utf-8")
        print(f"Debugging HTML report generated at: {debug_output_path.resolve()}")
        print("Open this file in your browser to inspect the passages.")

    print(f"Processed documents saved to: {documents_output_path.resolve()}")
    print(f"Summary report saved to: {summary_output_path.resolve()}")

    return processed_documents


def _build_html_header() -> List[str]:
    return [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8' />",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0' />",
        "<title>Parsing Debug Output</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1, h2, h3 { color: #333; }",
        ".passage-container { border: 1px solid #ccc; border-radius: 5px; padding: 15px; margin-bottom: 20px; background: #f9f9f9; }",
        ".passage-headers { font-weight: bold; color: #0073e6; margin-bottom: 10px; }",
        ".passage-content { margin: 10px 0; font-size: 1.1em; color: #444; }",
        ".context { font-style: italic; color: #555; margin: 5px 0; }",
        ".metadata { font-size: 0.9em; color: #555; margin-top: 10px; }",
        ".metadata ul { list-style-type: none; padding: 0; margin: 0; }",
        ".metadata li { margin-bottom: 5px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Debugging Passages</h1>",
        "<p>Below are the passages extracted from the document(s). Review the headers, content, contexts, and metadata to verify correctness.</p>"
    ]


def _build_passages_section(passages: List[Passage], context_mode: List[str]) -> List[str]:
    section_content = []
    for idx, passage in enumerate(passages):
        #final_text = assemble_passage_text(passage, context_mode)

        section_content.append("<div class='passage-container'>")

        # Headers Section
        headers_text = " > ".join(passage.headers) if passage.headers else "No Headers"
        section_content.append(f"<div class='passage-headers'>Passage {idx + 1} Headers: {headers_text}</div>")

        # Content Section
        section_content.append(f"<div class='passage-content'><strong>Content:</strong> {passage.content}</div>")

        # Contexts Section
        section_content.append("<div class='context'><strong>Context Before:</strong> ")
        section_content.append(passage.ctx_before or "None")
        section_content.append("</div>")
        section_content.append("<div class='context'><strong>Context After:</strong> ")
        section_content.append(passage.ctx_after or "None")
        section_content.append("</div>")
        section_content.append("<div class='context'><strong>Page Title:</strong> ")
        section_content.append(passage.page_title or "None")
        section_content.append("</div>")
        section_content.append("<div class='context'><strong>Headers:</strong> ")
        section_content.append(str(passage.headers) or "None")
        section_content.append("</div>")

        # Metadata Section
        section_content.append("<div class='metadata'><strong>Metadata:</strong><ul>")
        section_content.append(f"<li><strong>Passage ID:</strong> {passage.passage_id}</li>")
        section_content.append(f"<li><strong>Origin Identifier:</strong> {passage.space}</li>")
        section_content.append(f"<li><strong>Is Table:</strong> {passage.is_table}</li>")
        section_content.append(f"<li><strong>Is Table Row:</strong> {passage.is_table_row}</li>")
        section_content.append(f"<li><strong>Is List:</strong> {passage.is_list}</li>")
        if passage.attachment:
            section_content.append(f"<li><strong>Attachment:</strong> {passage.attachment}</li>")
        section_content.append("</ul></div>")

        section_content.append("</div>")

    section_content.append("</div>")
    return section_content


def generate_page_debugging_content(document: Document, passages: List[Passage], context_mode: List[str]) -> List[str]:
    page_content = [_build_document_header(document)]
    page_content.extend(_build_passages_section(passages, context_mode))
    return page_content


def _build_document_header(document: Document) -> str:
    return (
        f"<h2>Document ID: {document.id}</h2>"
        f"<p><strong>Document Title:</strong> {document.title}</p>"
        f"<p><strong>Document URL:</strong> {document.url}</p>"
        "<hr />"
        "<h3>Extracted Passages</h3>"
        "<div class='document-passages'>"
    )
