from typing import List, Dict, Any

from preprocessing.model import Passage


class ContentProcessor:
    """
    Manages the accumulation of text content until it can be formed into a Passage.
    """
    def __init__(self):
        self.current_content = []

    def add_text(self, text: str):
        if text:
            self.current_content.append(text)

    def clear_content(self):
        self.current_content.clear()

    def finalize_passage(self, current_headers: List[Dict[str, Any]], document_id: str, passage_count: int) -> Passage:
        if not self.current_content:
            return None
        passage = Passage(
            headers=[h['text'] for h in current_headers],
            content=' '.join(self.current_content).strip(),
            passage_id=f"{document_id}-{passage_count}",
        )
        self.clear_content()

        return passage
