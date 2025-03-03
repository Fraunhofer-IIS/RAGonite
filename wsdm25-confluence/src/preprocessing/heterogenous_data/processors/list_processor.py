from typing import List, Dict, Any

from bs4 import Tag

from preprocessing.model import Passage


class ListProcessor:
    def process(self, element: Tag, current_headers: List[Dict[str, Any]], document_id: str, passage_count: int) -> Passage:
        list_items = []
        if element.name == 'ul':
            items = element.find_all('li')
            for item in items:
                text = item.get_text(separator=' ', strip=True)
                list_items.append(f"- {text}")
        elif element.name == 'ol':
            items = element.find_all('li')
            for index, item in enumerate(items, start=1):
                text = item.get_text(separator=' ', strip=True)
                list_items.append(f"{index}) {text}")
        else:
            items = element.find_all('li')  # Only li supported currently
            for item in items:
                text = item.get_text(separator=' ', strip=True)
                list_items.append(f"- {text}")

        list_content = ' '.join(list_items)
        passage = Passage(
            headers=[h['text'] for h in current_headers],
            content=list_content.strip(),
            is_list=True,
            passage_id=f"{document_id}-{passage_count}"
        )
        return passage
