from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

Embedding = List[float]


class Table(BaseModel):
    headers: List[str]
    rows: List[List[str]]


class TableAttachment(BaseModel):
    id: str
    row: Optional[int] = None
    table: Optional[Table] = None
    type: Literal["table"] = "table"


Attachment = Union[TableAttachment]


class Document(BaseModel):
    id: str
    title: str
    content: str
    url: str
    embedding: Optional[Embedding] = None
    score: float = 0
    metadata: Optional[Dict[str, Any]] = None
    attachment: Optional[Attachment] = None
