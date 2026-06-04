"""RAG answer generation from hybrid retrieval results.

Pipeline:

    user query
      -> p3_hybrid_retrieval.hybrid_search()
      -> context built from retrieved chunks
      -> OpenAI or Gemini model answer

Set ``OPENAI_API_KEY`` or ``GEMINI_API_KEY`` in the environment or in a local
``.env`` file before running generation. Retrieval-only mode remains available through
``p3_hybrid_retrieval.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from p3_hybrid_retrieval import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, DEFAULT_HF_CACHE_DIR, DEFAULT_MODEL
from p3_hybrid_retrieval import repair_metadata, repair_mojibake
from p3_hybrid_retrieval import hybrid_search, run_search


DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


SYSTEM_INSTRUCTIONS = """Tu es un assistant culinaire pour un système RAG.
Réponds uniquement à partir du contexte fourni.
Si le contexte ne contient pas l'information nécessaire, dis clairement que tu ne sais pas.
Ne fabrique pas d'ingrédients, d'étapes, de quantités ou de sources.
Réponds dans la même langue que la question quand c'est naturel.
Mentionne les recettes utilisées et ajoute une courte section "Sources" à la fin."""


def build_context(results: list[dict[str, Any]], *, max_chars: int = 7000) -> str:
    """Convert retrieved chunks into a compact source-labeled context."""
    blocks: list[str] = []
    used_chars = 0
    for i, result in enumerate(results, start=1):
        metadata = repair_metadata(result.get("metadata", {}))
        text = repair_mojibake(result.get("text", ""))
        block = (
            f"[Source {i}]\n"
            f"Recipe: {metadata.get('recipe_name', 'unknown')}\n"
            f"Section: {metadata.get('section_type', 'unknown')}\n"
            f"Chunk ID: {result.get('id')}\n"
            f"URL: {metadata.get('source_url', '')}\n"
            f"Retriever: {','.join(result.get('sources', [result.get('retriever', '')]))}\n"
            f"Score: {result.get('score', 0):.4f}\n"
            f"Content:\n{text}"
        )
        if used_chars + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    return (
        "Contexte récupéré:\n"
        f"{context}\n\n"
        "Question utilisateur:\n"
        f"{query}\n\n"
        "Réponse:"
    )


def generate_answer(
    query: str,
    *,
    retrieval_top_k: int = 5,
    retrieval_mode: str = "hybrid",
    keyword_k: int = 20,
    vector_k: int = 20,
    llm_model: str = DEFAULT_LLM_MODEL,
    provider: str = "openai",
    context_max_chars: int = 7000,
    temperature: float = 0.2,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_MODEL,
    hf_cache_dir: Path = DEFAULT_HF_CACHE_DIR,
) -> dict[str, Any]:
    """Retrieve context and generate an answer with an OpenAI or Gemini model."""
    load_dotenv()

    if retrieval_mode == "hybrid":
        retrieval = hybrid_search(
            query,
            top_k=retrieval_top_k,
            keyword_k=keyword_k,
            vector_k=vector_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
            model_name=embedding_model,
            cache_dir=hf_cache_dir,
        )
    else:
        retrieval = run_search(
            query,
            mode=retrieval_mode,
            top_k=retrieval_top_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
            model_name=embedding_model,
            cache_dir=hf_cache_dir,
        )
    context = build_context(retrieval["results"], max_chars=context_max_chars)

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return {
            "query": query,
            "answer": None,
            "error": "OPENAI_API_KEY is not set. Add it to the environment or to a local .env file.",
            "context": context,
            "retrieval": retrieval,
        }
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        return {
            "query": query,
            "answer": None,
            "error": "GEMINI_API_KEY is not set. Add it to the environment or to a local .env file.",
            "context": context,
            "retrieval": retrieval,
        }

    if provider == "gemini":
        answer = _generate_with_gemini(
            prompt=build_prompt(query, context),
            model=llm_model,
            temperature=temperature,
        )
        return {
            "query": query,
            "answer": answer,
            "model": llm_model,
            "provider": provider,
            "context": context,
            "retrieval": retrieval,
        }

    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=llm_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=build_prompt(query, context),
        temperature=temperature,
    )

    return {
        "query": query,
        "answer": response.output_text,
        "model": llm_model,
        "provider": provider,
        "context": context,
        "retrieval": retrieval,
    }


def _generate_with_gemini(*, prompt: str, model: str, temperature: float) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            temperature=temperature,
        ),
    )
    return response.text or ""


def _print_sources(results: list[dict[str, Any]]) -> None:
    print("\nRetrieved sources:")
    for i, result in enumerate(results, start=1):
        metadata = repair_metadata(result.get("metadata", {}))
        sources = ",".join(result.get("sources", [result.get("retriever", "")]))
        print(
            f"{i}. {metadata.get('recipe_name', '')} "
            f"[{metadata.get('section_type', '')}] score={result.get('score', 0):.4f} source={sources}"
        )
        print(f"   id: {result.get('id')}")
        if metadata.get("source_url"):
            print(f"   url: {metadata['source_url']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Generate a RAG answer from hybrid retrieval.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--provider", choices=("openai", "gemini"), default="gemini")
    parser.add_argument("--model", default=None, help="Generation model")
    parser.add_argument("--retrieval-mode", choices=("keyword", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5, help="Number of fused chunks used as context")
    parser.add_argument("--keyword-k", type=int, default=20)
    parser.add_argument("--vector-k", type=int, default=20)
    parser.add_argument("--context-max-chars", type=int, default=7000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--json", action="store_true", help="Print full JSON payload")
    parser.add_argument("--show-context", action="store_true", help="Print the retrieved context")
    args = parser.parse_args()

    payload = generate_answer(
        args.query,
        retrieval_top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        keyword_k=args.keyword_k,
        vector_k=args.vector_k,
        llm_model=args.model or (DEFAULT_GEMINI_MODEL if args.provider == "gemini" else DEFAULT_LLM_MODEL),
        provider=args.provider,
        context_max_chars=args.context_max_chars,
        temperature=args.temperature,
        persist_directory=args.chroma_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        hf_cache_dir=args.hf_cache_dir,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if payload.get("error"):
        print(payload["error"])
        print("\nRetrieval worked; generation was skipped.")
    else:
        print(payload["answer"])

    _print_sources(payload["retrieval"]["results"])
    if args.show_context:
        print("\nContext:")
        print(payload["context"])


if __name__ == "__main__":
    main()
