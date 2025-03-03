from pathlib import Path
from typing import Optional, List, Dict, Any
from bs4 import Tag

from preprocessing.heterogenous_data.verbalization import verbalize_table, html_table_to_markdown, html_table_to_piped, table_to_json
from preprocessing.model import Passage, VerbalizerConfig
from models.data import TableAttachment
from preprocessing.utils import store_table, sanitize_passage_id


class TableProcessor:
    """
    Processes table elements, returning one or more Passages depending on granularity.
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path

    def process(self, element: Tag, current_headers: List[Dict[str, Any]], verbalizer_config: VerbalizerConfig, document_id: str, passage_count: int) -> List[Passage]:
        mode = verbalizer_config.mode[0]
        granularity = verbalizer_config.granularity

        if mode == 'verbalization':
            table_lines = verbalize_table(element)
            table_content = '\n'.join(line.strip() for line in table_lines)
        elif mode == 'piped':
            table_content, table_lines = html_table_to_piped(str(element), truncate_cells=False)
        elif mode == 'markdown':
            table_content, table_lines = html_table_to_markdown(str(element))
        elif mode == 'html':
            rows = element.find_all('tr')
            table_lines = [str(row) for row in rows]
            table_content = str(element)
        elif mode == 'plaintext':
            rows = element.find_all('tr')
            table_lines = [row.get_text(separator=' ', strip=True) for row in rows]
            table_content = element.get_text(separator=' ', strip=True)
        else:
            rows = element.find_all('tr')
            table_lines = [row.get_text(separator=' ', strip=True) for row in rows]
            table_content = element.get_text(separator=' ', strip=True)

        passage_headers = [h['text'] for h in current_headers]
        table_passages = []

        passage_id = sanitize_passage_id(f"{document_id}-{passage_count}")

        if any(g in ["element", "all"] for g in granularity):
            for line_count, line in enumerate(table_lines):
                row_id = passage_id + f"-{line_count}"
                psg = Passage(
                    headers=passage_headers,
                    content=line.strip(),
                    is_table_row=True,
                    passage_id=row_id,
                    attachment=self.create_attachment(passage_id, table_wise=False),
                )

                table_passages.append(psg)

        if any(g in ["entity", "all"] for g in granularity):
            psg = Passage(
                headers=passage_headers,
                content=table_content,
                is_table=True,
                passage_id=passage_id,
                attachment=self.create_attachment(passage_id, table_wise=True),
            )

            table_passages.append(psg)

        full_table, rows = html_table_to_piped(str(element))
        if self.db_path:
            store_table(self.db_path, passage_id, full_table, table_json=table_to_json(element))

        return table_passages

    @staticmethod
    def create_attachment(passage_id: str, table_wise: bool) -> Optional[TableAttachment]:
        if not passage_id:
            return None

        numeric_parts = [part for part in passage_id.split("-") if part.isdigit()]

        if table_wise:
            # Table-level attachment (requires at least 2 numeric parts: document_id and passage_count)
            if len(numeric_parts) >= 2:
                return TableAttachment(id=f"{numeric_parts[0]}-{numeric_parts[1]}", row=None)
        else:
            # Row-level attachment (requires at least 3 numeric parts: document_id, passage_count, and row)
            if len(numeric_parts) >= 3:
                try:
                    row_number = int(numeric_parts[2])
                    return TableAttachment(id=f"{numeric_parts[0]}-{numeric_parts[1]}", row=row_number)
                except ValueError:
                    pass

        return None



