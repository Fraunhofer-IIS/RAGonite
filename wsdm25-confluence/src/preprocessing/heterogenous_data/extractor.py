from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup, Tag

from preprocessing.heterogenous_data.processors.content_processor import ContentProcessor
from preprocessing.heterogenous_data.processors.header_processor import HeaderProcessor
from preprocessing.heterogenous_data.processors.list_processor import ListProcessor
from preprocessing.heterogenous_data.processors.table_processor import TableProcessor
from preprocessing.model import Passage
from preprocessing.heterogenous_data.contextualization import Contextualizer

from preprocessing.model import VerbalizerConfig
from models.data import Document

header_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']


class PassageExtractor:
    def __init__(self, verbalizer_config: VerbalizerConfig, document: Document, modalities: List[str], db_path: Optional[Path] = None):
        self.verbalizer_config = verbalizer_config
        self.document = document

        self.header_processor = HeaderProcessor()
        self.list_processor = ListProcessor()
        self.table_processor = TableProcessor(db_path=db_path)
        self.content_processor = ContentProcessor()

        self.contextualizer = None

        self.allowed_modalities = set(modalities)
        if "all" in self.allowed_modalities:
            self.allowed_modalities = {"passage", "table", "list"}
        elif "none" in self.allowed_modalities:
            self.allowed_modalities = {"passage"}  # Include only passages (no tables or lists)

    def extract_passages(self, soup: BeautifulSoup) -> List[Passage]:
        passages = []
        current_headers = []
        processed_nodes = set()

        elements = list(soup.descendants)
        total_elements = len(elements)

        self.contextualizer = Contextualizer(elements, self.verbalizer_config.context_mode)

        def is_inside_special_tag(el, tags=('table', 'ul', 'ol')):
            parent = getattr(el, 'parent', None)
            while parent is not None:
                if getattr(parent, 'name', None) in tags:
                    return True
                parent = parent.parent
            return False

        for idx, element in enumerate(elements):
            if element in processed_nodes:
                continue

            if isinstance(element, Tag):
                if element.name in header_tags:
                    # If we have accumulated content before this header, finalize that passage first
                    if self.content_processor.current_content:
                        passage = self.content_processor.finalize_passage(current_headers, document_id=self.document.id, passage_count=len(passages))
                        if passage:
                            self.contextualizer.collect_contexts(passage, idx, current_headers, self.document.title,
                                                                 is_list_or_table=False)
                            passages.append(passage)

                    self.header_processor.process(element, current_headers)

                elif element.name == 'table':
                    if "table" in self.allowed_modalities:
                        table_passages = self.table_processor.process(element, current_headers, self.verbalizer_config,
                                                                      document_id=self.document.id, passage_count=len(passages))
                        for psg in table_passages:
                            self.contextualizer.collect_contexts(psg, idx, current_headers, self.document.title,
                                                                 is_list_or_table=True)
                        passages.extend(table_passages)

                    # Mark table and descendants as processed, regardless of inclusion
                    processed_nodes.add(element)
                    for desc in element.descendants:
                        processed_nodes.add(desc)

                    self.content_processor.clear_content()

                elif element.name in ['ul', 'ol']:
                    if "list" in self.allowed_modalities:
                        list_passage = self.list_processor.process(element, current_headers,
                                                                   document_id=self.document.id, passage_count=len(passages))
                        self.contextualizer.collect_contexts(list_passage, idx, current_headers, self.document.title,
                                                             is_list_or_table=True)
                        passages.append(list_passage)

                    # Mark list and descendants as processed, regardless of inclusion
                    processed_nodes.add(element)
                    for desc in element.descendants:
                        processed_nodes.add(desc)

                    self.content_processor.clear_content()

                else:
                    # Generic content element: only add text if it's not already processed or within a processed special tag
                    if element not in processed_nodes and not is_inside_special_tag(element):
                        text = element.get_text(separator=' ', strip=True)
                        if text:
                            self.content_processor.add_text(text)

        # Finalize any remaining content after processing all elements
        if self.content_processor.current_content:
            passage = self.content_processor.finalize_passage(current_headers, document_id=self.document.id, passage_count=len(passages))
            if passage:
                self.contextualizer.collect_contexts(passage, total_elements - 1, current_headers, self.document.title,
                                                     is_list_or_table=False)
                passages.append(passage)

        return passages


def parse_html_content(html_content: str) -> BeautifulSoup:
    return BeautifulSoup(html_content, 'html.parser')
