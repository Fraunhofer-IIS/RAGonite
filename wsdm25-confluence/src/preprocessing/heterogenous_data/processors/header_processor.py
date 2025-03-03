from typing import List, Dict, Any

from bs4 import Tag


class HeaderProcessor:
    """
    Handles updating the current header structure when a new header tag is encountered.
    """
    def process(self, element: Tag, current_headers: List[Dict[str, Any]]):
        header_level = int(element.name[1])
        header_text = element.get_text(separator=' ', strip=True)

        # Adjust current_headers to maintain a hierarchical structure
        while current_headers and current_headers[-1]['level'] >= header_level:
            current_headers.pop()

        current_headers.append({'level': header_level, 'text': header_text})
