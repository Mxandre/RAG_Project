import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib


def jsonl_to_documents(
    jsonl_path: str | Path,
    *,
    text_field: str = "text",
    metadata_field: str = "metadata",
    id_field: str = "id",
) -> list[Document]:
    documents: list[Document] = []
    path = Path(jsonl_path)

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue

            item: dict[str, Any] = json.loads(line)
            text = item.get(text_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Ligne {line_no} : le champ '{text_field}' est manquant ou vide")

            raw_metadata = item.get(metadata_field, {})
            if raw_metadata is None:
                raw_metadata = {}
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"Ligne {line_no} : le champ '{metadata_field}' doit être un objet")

            metadata = {
                **raw_metadata,
                "id": item.get(id_field),
                "source": raw_metadata.get("source", str(path)),
                "line_no": line_no,
            }

            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def build_chroma_vectorstore(
    jsonl_path: str | Path,
    embeddings: Embeddings,
    *,
    persist_directory: str | Path,
    collection_name: str = "rag_documents",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = jsonl_to_documents(jsonl_path)
    chunks = splitter.split_documents(documents)
    chunk_ids = [_make_chunk_id(chunk) for chunk in chunks]
    for chunk, chunk_id in zip(chunks, chunk_ids):
        chunk.metadata["chunk_id"] = chunk_id
    return Chroma.from_documents(
        documents=chunks,
        ids=chunk_ids,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def _make_chunk_id(chunk):
    raw = f"{chunk.metadata.get('id','')}::{chunk.page_content}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return digest
