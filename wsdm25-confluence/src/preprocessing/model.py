from pathlib import Path
from typing import Literal, Optional, List, Union
from dataclasses import dataclass, field
from models.data import Document, Attachment


@dataclass
class VerbalizerConfig:
    mode: List[Literal["verbalization", "piped", "markdown", "plaintext", "html"]] = field(default_factory=lambda: ["verbalization"])

    # Granularity is the "scope" of what should be handled as one package to embed to the DB.
    # Element could be a table-row, or a single property in a graph-db
    # Entity could be a complete table, or an entire entity in a graph-db
    # "All" will create multiple variants and store them in parallel.
    granularity: List[Literal["element", "entity", "all"]] = field(default_factory=lambda: ["element"])

    # Whether to keep the original content after verbalization and store it alongside the verbalizations.
    embed_original_content: bool = False
    context_mode: List[Literal["all", "none", "entity_title", "preceding_heading", "page_title", "ctx_before", "ctx_after"]] = field(default_factory=lambda: ["all"])


@dataclass
class MultiModalConfig:
    verbalizer_config: Optional[VerbalizerConfig] = None
    modalities: List[Literal["all", "none", "table", "list"]] = field(default_factory=lambda: ["all"])


class VerbalizerDocument(Document):
    """
    Enriched Document holding additional meta-information computed during verbalization.
    Please note that as of now, this information do not get stored in the DB.
    It is only used for
    """
    content: Union[str, dict]
    raw: str
    space: Optional[str] = None  # Could be a confluence space key, or a graph node ID, etc.
    entity_descriptions: Optional[List[str]] = None  # Could be a list of table captions
    num_verbalized_entities: int = 0  # A counter how many verbalized entities are contained, e.g. number of tables


@dataclass
class Passage:
    passage_id: str = ""
    headers: List[str] = field(default_factory=list)
    space: str = ""
    content: str = ""
    entity_title: str = ""
    page_title: str = ""
    ctx_before: str = ""
    ctx_after: str = ""

    is_table: bool = False
    is_table_row: bool = False
    is_list: bool = False

    attachment: Optional[Attachment] = None