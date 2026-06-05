import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
            raw_text = item.get(text_field)
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(f"Line {line_no} missing non-empty '{text_field}'")

            raw_metadata = item.get(metadata_field, {})
            if raw_metadata is None:
                raw_metadata = {}
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"Line {line_no} field '{metadata_field}' must be an object")

            metadata = {
                **raw_metadata,
                "id": item.get(id_field),
                "source": raw_metadata.get("source", str(path)),
                "line_no": line_no,
            }
            page_content = _build_page_content(raw_text, metadata)

            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def build_chroma_vectorstore(
    jsonl_path: str | Path,
    embeddings: Embeddings,
    *,
    persist_directory: str | Path,
    collection_name: str = "rag_documents",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
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


def _build_page_content(text: str, metadata: dict[str, Any]) -> str:
    recipe_name = metadata.get("recipe_name", "")
    section_type = metadata.get("section_type", "")
    servings = metadata.get("servings", "")
    section_label = {
        "ingredients": "Ingredients de la recette",
        "steps": "Etapes de preparation",
    }.get(section_type, section_type)

    parts = [
        f"Recette: {recipe_name}" if recipe_name else "",
        f"Section: {section_label}" if section_label else "",
        f"Portions: {servings}" if servings else "",
        "",
        text,
    ]
    return "\n".join(part for part in parts if part)


def _make_chunk_id(chunk: Document) -> str:
    raw = f"{chunk.metadata.get('id', '')}::{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
