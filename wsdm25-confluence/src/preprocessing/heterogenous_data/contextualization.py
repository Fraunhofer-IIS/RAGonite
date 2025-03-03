from typing import Dict, Any, Iterable, List
from bs4 import PageElement, Tag
from preprocessing.model import Passage


class Contextualizer:
    def __init__(self, elements: List[Iterable[PageElement]], mode: List[str], special_processing_tags: Dict[str, Any] = None):
        self.elements = elements
        self.special_processing_tags = special_processing_tags or {'table', 'ul', 'ol'}
        self.mode = mode

    def collect_contexts(self, passage: Passage, idx: int, current_headers: List[Dict[str, Any]], document_title: str, is_list_or_table: bool):
        passage.page_title = document_title

        # Determine contexts based on whether this is a list/table or a normal passage.
        # This was necessary because we are treating them specially by processing a full list or table at once.
        # However, as we are still looping over the entire DOM, it would be possible that lists or tables
        # add parts of themselves as context.
        if is_list_or_table:
            # For lists/tables, we want the nearest suitable non-special neighbors, or if special, extract last/first row/li
            passage.ctx_before = self._find_previous_non_special_context(idx, current_headers)
            passage.ctx_after = self._find_next_non_special_context(idx, current_headers)
        else:
            # For normal passages, just take immediate neighbors if suitable
            passage.ctx_before = self._safe_get_context(idx - 1, current_headers, direction="prev")
            passage.ctx_after = self._safe_get_context(idx + 1, current_headers, direction="next")

    def _get_element_text(self, el: Tag) -> str:
        if hasattr(el, 'get_text'):
            return el.get_text(strip=True)
        return str(el).strip()

    def _extract_list_or_table_text(self, el: Tag, direction: str) -> str:
        """
        Extract specific text from a table or a list depending on the direction.
        It would be insane to add a full table as context for every "element"/row-wise document.

        direction = "prev" means we should get the last row/li
        direction = "next" means we should get the first row/li
        """
        name = getattr(el, 'name', '')
        if name == 'table':
            rows = el.find_all('tr')
            if rows:
                target_row = rows[-1] if direction == "prev" else rows[0]
                return target_row.get_text(strip=True) if target_row else ''
        elif name in ['ul', 'ol']:
            items = el.find_all('li')
            if items:
                target_item = items[-1] if direction == "prev" else items[0]
                return target_item.get_text(strip=True) if target_item else ''
        # If it's not a recognized special tag or no items found
        return self._get_element_text(el)

    def _find_previous_non_special_context(self, idx: int, current_headers: List[Dict[str, Any]]) -> str:
        # Scan backwards for a suitable context element
        for i in range(idx - 1, -1, -1):
            el = self.elements[i]
            name = getattr(el, 'name', '')

            # If inside a special tag, try extracting the last row/item
            if name in self.special_processing_tags:
                text = self._extract_list_or_table_text(el, direction="prev")
                if text and not self._is_header_duplication(text, current_headers):
                    return text
        return ''

    def _find_next_non_special_context(self, idx: int, current_headers: List[Dict[str, Any]]) -> str:
        # Scan forwards for a suitable context element
        for i in range(idx + 1, len(self.elements)):
            el = self.elements[i]
            name = getattr(el, 'name', '')

            # If inside a special tag, try extracting the first row/item
            if name in self.special_processing_tags:
                text = self._extract_list_or_table_text(el, direction="next")
                if text and not self._is_header_duplication(text, current_headers):
                    return text
        return ''

    def _safe_get_context(self, idx: int, current_headers: List[Dict[str, Any]], direction: str = "next") -> str:
        # Returns the text of a neighboring element if it exists. If it's a special tag,
        # handle according to direction.
        if 0 <= idx < len(self.elements):
            el = self.elements[idx]
            name = getattr(el, 'name', '')
            if name in self.special_processing_tags:
                text = self._extract_list_or_table_text(el, direction=direction)
            else:
                text = self._get_element_text(el)

            if text and not self._is_header_duplication(text, current_headers):
                return text
        return ''

    @staticmethod
    def _is_header_duplication(text: str, current_headers: List[Dict[str, Any]]) -> bool:
        """
        Checks if the given text is effectively just repeating a heading that's already used in the passage.
        This helps avoid duplication where the context before is the same as one of the headings.
        """
        header_texts = [h['text'] for h in current_headers]
        return text in header_texts


def assemble_passage_text(passage: Passage, context_mode: List[str]) -> str:
    if 'all' in context_mode:
        context_mode = ["entity_title", "preceding_heading", "page_title", "ctx_before", "ctx_after"]

    components = []
    if 'page_title' in context_mode and passage.page_title:
        components.append(passage.page_title)
    if 'preceding_heading' in context_mode and passage.headers:
        components.append(" ".join(passage.headers))
    if 'entity_title' in context_mode and passage.entity_title:
        components.append(passage.entity_title)
    if 'ctx_before' in context_mode and passage.ctx_before:
        components.append(passage.ctx_before)
    components.append(passage.content)
    if 'ctx_after' in context_mode and passage.ctx_after:
        components.append(passage.ctx_after)
    final_text = '\n'.join(components)

    passage.content = final_text
    return passage
