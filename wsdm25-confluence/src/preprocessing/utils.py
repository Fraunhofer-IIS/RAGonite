import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Optional, Dict, Any, List
from preprocessing.model import Passage


def recursive_to_dict(obj):
    if isinstance(obj, dict):
        return {key: recursive_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, list):
        return [recursive_to_dict(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return {key: recursive_to_dict(value) for key, value in vars(obj).items()}
    elif hasattr(obj, '__slots__'):
        return {key: recursive_to_dict(getattr(obj, key)) for key in obj.__slots__}
    else:
        return obj


def save_config_as_json(db_filepath: Path, config: dict):
    json_filepath = os.path.join(db_filepath, f"multi_modal_config.json")

    serializable_config = recursive_to_dict(config)

    with open(json_filepath, 'w') as json_file:
        json.dump(serializable_config, json_file, indent=4)

    print(f"Configuration saved to {json_filepath}")


def truncate(text, max_length=10, buffer=5):
    """ Truncates text to a maximum length, extending to complete the current word if it goes slightly over. """
    if len(text) <= max_length:
        return text
    # Look for the next space after the maximum length limit to finish the current word.
    end = text.find(' ', max_length)
    if end == -1 or end > max_length + buffer:
        # No space found within the buffer range or no space at all, truncate at max_length
        return text[:max_length] + "..."
    return text[:end] + "..."


def store_table(db_path: Path, table_id: str, table_html, table_json: Optional[Dict[str, Any]] = None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS tables
                 (table_id TEXT PRIMARY KEY, table_html TEXT, table_json TEXT)''')

    # Overwrite existing table if exists
    c.execute('SELECT COUNT(*) FROM tables WHERE table_id = ?', (table_id,))
    if c.fetchone()[0] > 0:
        c.execute('DELETE FROM tables WHERE table_id = ?', (table_id,))

    json_data = json.dumps(table_json) if table_json else None
    c.execute('INSERT INTO tables (table_id, table_html, table_json) VALUES (?, ?, ?)', (table_id, table_html, json_data))

    conn.commit()
    conn.close()


def generate_heterogenous_processing_summary_from_passages(passages: List[Passage], document_count: Optional[int] = None) -> str:
    spaces = set()
    unique_pages = set()
    pages_per_origin = defaultdict(int)
    tables_per_origin = defaultdict(int)

    total_tables = 0
    total_words = 0
    total_lists = 0
    pages_with_tables = set()
    pages_with_lists = set()
    pages_with_passages = set()
    pages_with_passage_list_table = set()

    table_text_lengths = []
    passage_word_lengths = []
    list_word_lengths = []

    unique_tables = set()  # To track distinct tables by `document_id` and `passage_count`
    unique_lists = set()  # To track distinct lists by `document_id` and `passage_count`

    page_feature_tracker = defaultdict(lambda: {"has_table": False, "has_list": False, "has_passage": False})

    for passage in passages:
        space = passage.space
        spaces.add(space)

        page_title = passage.page_title
        unique_pages.add(page_title)
        pages_per_origin[space] += 1

        # Count words in passage content
        words_in_passage = len(passage.content.split())
        total_words += words_in_passage
        passage_word_lengths.append(words_in_passage)

        if passage.is_table:
            unique_table_id = f"{passage.passage_id}"  # `document_id-passage_count` for unique tables
            if unique_table_id not in unique_tables:
                unique_tables.add(unique_table_id)
                total_tables += 1
                pages_with_tables.add(page_title)
                table_text_lengths.append(words_in_passage)
                page_feature_tracker[page_title]["has_table"] = True

        if passage.is_list:
            unique_list_id = f"{passage.passage_id}"  # `document_id-passage_count` for unique lists
            if unique_list_id not in unique_lists:
                unique_lists.add(unique_list_id)
                total_lists += 1
                pages_with_lists.add(page_title)
                list_word_lengths.append(words_in_passage)
                page_feature_tracker[page_title]["has_list"] = True

        if not passage.is_table and not passage.is_list and passage.content.strip() and not page_feature_tracker[page_title]["has_passage"]:
            pages_with_passages.add(page_title)
            page_feature_tracker[page_title]["has_passage"] = True

    for page_title, features in page_feature_tracker.items():
        if features["has_table"] and features["has_list"] and features["has_passage"]:
            pages_with_passage_list_table.add(page_title)

    total_passages = len(passages)
    only_passages = total_passages - total_lists - total_tables

    median_table_text_lengths = median(table_text_lengths) if table_text_lengths else 0
    median_passage_word_length = median(passage_word_lengths) if passage_word_lengths else 0
    median_list_word_length = median(list_word_lengths) if list_word_lengths else 0

    summary_lines = [
        10 * "=" + " SUMMARY " + 10 * "=",
        f"Processed Pages in Total: {len(unique_pages)}",
        f"Unique Origins: {', '.join(spaces)}",
        "Pages Processed Per Origin:"
    ]
    summary_lines.extend([f"  {origin}: {count}" for origin, count in pages_per_origin.items()])
    summary_lines.append("Tables in Each Origin:")
    summary_lines.extend([f"  {origin}: {count}" for origin, count in tables_per_origin.items()])

    summary_lines.extend([
        f"Total Number of Tables Across All Data: {total_tables}",
        f"Number of Pages with Tables: {len(pages_with_tables)}",
        f"Total Number of Lists: {total_lists}",
        f"Number of Pages with Lists: {len(pages_with_lists)}",
        f"Number of Pages with Passages: {len(pages_with_passages)}",
        f"Number of Pages with Passages + Lists + Tables: {len(pages_with_passage_list_table)}",
        f"Total Number of Passages (including tables, lists): {total_passages}",
        f"Number of pure Passages: {only_passages}",
        f"Total Number of Words: {total_words}",
        f"Median Word Length per Passage: {median_passage_word_length}",
        f"Median Table Text Length (words): {median_table_text_lengths}",
        f"Median List Size (words): {median_list_word_length}",
        f"Num Pages with Text+Lists: {len(pages_with_lists)}",
        f"Num Pages with Text+Tables: {len(pages_with_tables)}",
        f"Num Pages with Tables+Lists: {len(pages_with_passage_list_table)}",
        10 * "=" + " END SUMMARY " + 10 * "=" + "\n",
        f"Total number of resulting documents due to mode (should match total number of passages): {document_count}" if document_count else "",
    ])

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    return summary_text


def sanitize_passage_id(passage_id):
    parts = passage_id.split('-')
    if len(parts) > 2:
        return '-'.join(parts[-2:])
    return passage_id
