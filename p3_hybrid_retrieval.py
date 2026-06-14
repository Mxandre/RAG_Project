"""Recherche hybride : index de mots-clés + recherche vectorielle.

Ce script combine :

- la recherche par mots-clés de ``p2_keyword_retrieval.py``
- la recherche vectorielle Chroma construite par ``data_process.py``

La fusion utilise Reciprocal Rank Fusion (RRF), robuste lorsque les scores BM25
et les distances vectorielles ne sont pas sur la même échelle.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from data_process import build_chroma_vectorstore
from p2_keyword_retrieval import CHUNKS_JSONL, keyword_search


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_COLLECTION = "recipes"
DEFAULT_CHROMA_DIR = Path("chroma_db")
DEFAULT_EVAL = Path("data") / "keyword_eval_queries.jsonl"
DEFAULT_HF_CACHE_DIR = Path(".hf_cache")
DEFAULT_SEARCH_CACHE_SIZE = 128
DEFAULT_SEARCH_CACHE_TTL_SECONDS = 600
DEFAULT_SEMANTIC_CACHE_THRESHOLD = 0.97


_SEARCH_CACHE: OrderedDict[tuple[Any, ...], tuple[float, dict[str, Any]]] = OrderedDict()


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake such as 'Ã©' -> 'é'."""
    if not isinstance(text, str) or "Ã" not in text:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def repair_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: repair_mojibake(value) if isinstance(value, str) else value for key, value in metadata.items()}


@lru_cache(maxsize=4)
def make_embeddings(model_name: str = DEFAULT_MODEL, cache_dir: Path = DEFAULT_HF_CACHE_DIR):
    """Crée la fonction d'embedding HuggingFace à la demande."""
    from langchain_huggingface import HuggingFaceEmbeddings

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((cache_dir / "transformers").resolve()))
    return HuggingFaceEmbeddings(
        model=model_name,
        cache_folder=str(cache_dir),
        model_kwargs={"local_files_only": True},
    )


@lru_cache(maxsize=8)
def load_vectorstore(
    *,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_HF_CACHE_DIR,
):
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=make_embeddings(model_name, cache_dir),
        collection_name=collection_name,
    )


def clear_retrieval_caches() -> None:
    """Clear in-memory embedding, vectorstore, and query-result caches."""
    _SEARCH_CACHE.clear()
    make_embeddings.cache_clear()
    load_vectorstore.cache_clear()


def rebuild_vectorstore(
    *,
    jsonl_path: Path = CHUNKS_JSONL,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_HF_CACHE_DIR,
    reset: bool = False,
):
    clear_retrieval_caches()
    if reset and persist_directory.exists():
        shutil.rmtree(persist_directory)

    return build_chroma_vectorstore(
        jsonl_path=jsonl_path,
        embeddings=make_embeddings(model_name, cache_dir),
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def vector_search(
    query: str,
    *,
    top_k: int = 5,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_HF_CACHE_DIR,
) -> dict[str, Any]:
    vectorstore = load_vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        model_name=model_name,
        cache_dir=cache_dir,
    )
    hits = vectorstore.similarity_search_with_score(query, k=top_k)

    results = []
    for rank, (doc, distance) in enumerate(hits, start=1):
        metadata = dict(doc.metadata)
        metadata = repair_metadata(metadata)
        source_id = metadata.get("id") or metadata.get("chunk_id") or f"vector::{rank}"
        text = repair_mojibake(doc.page_content)
        results.append(
            {
                "id": source_id,
                "score": 1.0 / (1.0 + float(distance)),
                "raw_distance": float(distance),
                "rank": rank,
                "text": text,
                "metadata": metadata,
                "matched_terms": [],
                "snippet": _snippet(text),
                "retriever": "vector",
            }
        )

    return {"query": query, "results": results}


def hybrid_search(
    query: str,
    *,
    top_k: int = 5,
    keyword_k: int = 20,
    vector_k: int = 20,
    rrf_k: int = 60,
    keyword_weight: float = 1.5,
    vector_weight: float = 0.7,
    rrf_weight: float = 0.4,
    score_weight: float = 1.0,
    coverage_weight: float = 0.2,
    multi_retriever_bonus: float = 0.05,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_HF_CACHE_DIR,
) -> dict[str, Any]:
    keyword_payload = keyword_search(query, top_k=keyword_k)
    vector_payload = vector_search(
        query,
        top_k=vector_k,
        persist_directory=persist_directory,
        collection_name=collection_name,
        model_name=model_name,
        cache_dir=cache_dir,
    )

    fused: dict[str, dict[str, Any]] = {}
    scores: defaultdict[str, float] = defaultdict(float)

    query_terms = set(keyword_payload.get("analysis", {}).get("corrected_keywords") or [])

    def normalized_scores(results: list[dict[str, Any]]) -> dict[str, float]:
        scores = [float(result.get("score", 0.0)) for result in results]
        if not scores:
            return {}
        low = min(scores)
        high = max(scores)
        out = {}
        for result in results:
            raw_score = float(result.get("score", 0.0))
            if high > low:
                out[result["id"]] = (raw_score - low) / (high - low)
            else:
                out[result["id"]] = 1.0 if raw_score > 0 else 0.0
        return out

    keyword_norms = normalized_scores(keyword_payload["results"])
    vector_norms = normalized_scores(vector_payload["results"])

    def add_results(results: list[dict[str, Any]], source: str, weight: float, norms: dict[str, float]) -> None:
        for rank, result in enumerate(results, start=1):
            result = {
                **result,
                "text": repair_mojibake(result.get("text", "")),
                "snippet": repair_mojibake(result.get("snippet", "")),
                "metadata": repair_metadata(result.get("metadata", {})),
            }
            result_id = result["id"]
            scores[result_id] += weight / (rrf_k + rank)
            if result_id not in fused:
                fused[result_id] = {
                    **result,
                    "retriever": "hybrid",
                    "keyword_score": 0.0,
                    "vector_score": 0.0,
                    "keyword_rank": None,
                    "vector_rank": None,
                    "sources": [],
                }
            fused[result_id]["sources"].append(source)
            fused[result_id][f"{source}_rank"] = rank
            fused[result_id][f"{source}_score"] = result.get("score", 0.0)
            fused[result_id][f"{source}_norm_score"] = norms.get(result_id, 0.0)

            # Les résultats mots-clés viennent du JSONL courant et gardent le
            # texte le plus propre. Les résultats vectoriels complètent le rang
            # et les entrées absentes, sans écraser ce texte.
            if source == "vector" and "keyword" not in fused[result_id]["sources"]:
                fused[result_id]["text"] = result["text"]
                fused[result_id]["snippet"] = result["snippet"]
                fused[result_id]["metadata"] = {**fused[result_id].get("metadata", {}), **result["metadata"]}

    add_results(keyword_payload["results"], "keyword", keyword_weight, keyword_norms)
    add_results(vector_payload["results"], "vector", vector_weight, vector_norms)

    results = []
    for result_id, result in fused.items():
        matched_terms = set(result.get("matched_terms") or [])
        coverage = len(matched_terms & query_terms) / max(len(query_terms), 1) if query_terms else 0.0
        retriever_count = len(set(result["sources"]))
        result["rrf_score"] = scores[result_id]
        result["term_coverage"] = coverage
        result["score"] = (
            rrf_weight * scores[result_id]
            + score_weight
            * (
                keyword_weight * result.get("keyword_norm_score", 0.0)
                + vector_weight * result.get("vector_norm_score", 0.0)
            )
            + coverage_weight * coverage
            + (multi_retriever_bonus if retriever_count > 1 else 0.0)
        )
        result["sources"] = sorted(set(result["sources"]))
        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": query,
        "analysis": keyword_payload.get("analysis", {}),
        "results": results[:top_k],
        "keyword_results": keyword_payload["results"],
        "vector_results": vector_payload["results"],
    }


def evaluate(
    eval_path: Path = DEFAULT_EVAL,
    *,
    mode: str = "hybrid",
    top_k_values: tuple[int, ...] = (1, 3, 5, 10),
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_HF_CACHE_DIR,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    max_k = max(top_k_values)
    metric_sums = {
        **{f"recall@{k}": 0.0 for k in top_k_values},
        **{f"precision@{k}": 0.0 for k in top_k_values},
        **{f"hit@{k}": 0.0 for k in top_k_values},
        "mrr": 0.0,
    }
    timings = []
    details = []

    for row in rows:
        query = row["query"]
        start = time.perf_counter()
        payload = run_search(
            query,
            mode=mode,
            top_k=max_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
            model_name=model_name,
            cache_dir=cache_dir,
        )
        timings.append(time.perf_counter() - start)
        results = payload["results"]
        relevant = [_is_relevant(result, row) for result in results]
        total_relevant = max(_expected_count(row), 1)

        first_hit_rank = None
        for index, is_hit in enumerate(relevant, start=1):
            if is_hit:
                first_hit_rank = index
                break
        if first_hit_rank:
            metric_sums["mrr"] += 1.0 / first_hit_rank

        for k in top_k_values:
            top_hits = sum(relevant[:k])
            metric_sums[f"hit@{k}"] += 1.0 if top_hits else 0.0
            metric_sums[f"precision@{k}"] += top_hits / k
            metric_sums[f"recall@{k}"] += min(top_hits / total_relevant, 1.0)

        details.append(
            {
                "query": query,
                "expected": _expected_label(row),
                "first_hit_rank": first_hit_rank,
                "top": [
                    {
                        "rank": i,
                        "hit": relevant[i - 1],
                        "id": result["id"],
                        "recipe_name": result.get("metadata", {}).get("recipe_name"),
                        "score": result["score"],
                        "retriever": result["retriever"],
                        "sources": result.get("sources", [result["retriever"]]),
                    }
                    for i, result in enumerate(results, start=1)
                ],
            }
        )

    n = max(len(rows), 1)
    metrics = {name: value / n for name, value in metric_sums.items()}
    return {
        "mode": mode,
        "n_queries": len(rows),
        "metrics": metrics,
        "avg_time_ms": (sum(timings) / max(len(timings), 1)) * 1000,
        "details": details,
    }


def compare_modes(
    eval_path: Path = DEFAULT_EVAL,
    *,
    modes: tuple[str, ...] = ("keyword", "vector", "hybrid"),
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_HF_CACHE_DIR,
) -> dict[str, Any]:
    reports = {}
    for mode in modes:
        report = evaluate(
            eval_path,
            mode=mode,
            persist_directory=persist_directory,
            collection_name=collection_name,
            model_name=model_name,
            cache_dir=cache_dir,
        )
        reports[mode] = {
            "n_queries": report["n_queries"],
            "avg_time_ms": report["avg_time_ms"],
            "metrics": report["metrics"],
        }
    return {"eval_path": str(eval_path), "reports": reports}


def run_search(
    query: str,
    *,
    mode: str,
    top_k: int,
    persist_directory: Path,
    collection_name: str,
    model_name: str,
    cache_dir: Path = DEFAULT_HF_CACHE_DIR,
    keyword_k: int = 20,
    vector_k: int = 20,
    use_cache: bool = True,
) -> dict[str, Any]:
    if use_cache:
        cached = _get_cached_search(
            query=query,
            mode=mode,
            top_k=top_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
            model_name=model_name,
            cache_dir=cache_dir,
            keyword_k=keyword_k,
            vector_k=vector_k,
        )
        if cached is not None:
            return cached

    if mode == "keyword":
        payload = keyword_search(query, top_k=top_k)
    elif mode == "vector":
        payload = vector_search(
            query,
            top_k=top_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
            model_name=model_name,
            cache_dir=cache_dir,
        )
    elif mode == "hybrid":
        payload = hybrid_search(
            query,
            top_k=top_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
            model_name=model_name,
            cache_dir=cache_dir,
            keyword_k=keyword_k,
            vector_k=vector_k,
        )
    else:
        raise ValueError(f"Mode inconnu : {mode}")

    if use_cache:
        _store_cached_search(
            query=query,
            mode=mode,
            top_k=top_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
            model_name=model_name,
            cache_dir=cache_dir,
            keyword_k=keyword_k,
            vector_k=vector_k,
            payload=payload,
        )
    return payload


def _get_cached_search(
    *,
    query: str,
    mode: str,
    top_k: int,
    persist_directory: Path,
    collection_name: str,
    model_name: str,
    cache_dir: Path,
    keyword_k: int,
    vector_k: int,
) -> dict[str, Any] | None:
    now = time.time()
    key = _search_cache_key(
        query,
        mode,
        top_k,
        persist_directory,
        collection_name,
        model_name,
        cache_dir,
        keyword_k,
        vector_k,
    )
    cached = _SEARCH_CACHE.get(key)
    if cached and now - cached[0] <= DEFAULT_SEARCH_CACHE_TTL_SECONDS:
        _SEARCH_CACHE.move_to_end(key)
        payload = copy.deepcopy(cached[1])
        payload.setdefault("analysis", {})["cache"] = {"hit": True, "type": "exact"}
        return payload

    normalized = _normalize_cache_query(query)
    tokens = set(normalized.split())
    if tokens:
        for candidate_key, (created_at, candidate_payload) in reversed(_SEARCH_CACHE.items()):
            if now - created_at > DEFAULT_SEARCH_CACHE_TTL_SECONDS:
                continue
            if candidate_key[0:8] != key[0:8]:
                continue
            similarity = _jaccard(tokens, set(str(candidate_key[8]).split()))
            if similarity >= DEFAULT_SEMANTIC_CACHE_THRESHOLD:
                _SEARCH_CACHE.move_to_end(candidate_key)
                payload = copy.deepcopy(candidate_payload)
                payload["query"] = query
                payload.setdefault("analysis", {})["cache"] = {
                    "hit": True,
                    "type": "semantic",
                    "matched_query": candidate_key[9],
                    "similarity": similarity,
                }
                return payload
    return None


def _store_cached_search(
    *,
    query: str,
    mode: str,
    top_k: int,
    persist_directory: Path,
    collection_name: str,
    model_name: str,
    cache_dir: Path,
    keyword_k: int,
    vector_k: int,
    payload: dict[str, Any],
) -> None:
    key = _search_cache_key(
        query,
        mode,
        top_k,
        persist_directory,
        collection_name,
        model_name,
        cache_dir,
        keyword_k,
        vector_k,
    )
    _SEARCH_CACHE[key] = (time.time(), copy.deepcopy(payload))
    _SEARCH_CACHE.move_to_end(key)
    while len(_SEARCH_CACHE) > DEFAULT_SEARCH_CACHE_SIZE:
        _SEARCH_CACHE.popitem(last=False)


def _search_cache_key(
    query: str,
    mode: str,
    top_k: int,
    persist_directory: Path,
    collection_name: str,
    model_name: str,
    cache_dir: Path,
    keyword_k: int,
    vector_k: int,
) -> tuple[Any, ...]:
    normalized = _normalize_cache_query(query)
    return (
        mode,
        top_k,
        str(Path(persist_directory)),
        collection_name,
        model_name,
        str(Path(cache_dir)),
        keyword_k,
        vector_k,
        normalized,
        query,
    )


def _normalize_cache_query(query: str) -> str:
    return " ".join(query.casefold().strip().split())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _is_relevant(result: dict[str, Any], row: dict[str, Any]) -> bool:
    expected_ids = set(row.get("expected_chunk_ids") or [])
    expected_recipes = set(row.get("expected_recipes") or [])
    if row.get("expected_recipe"):
        expected_recipes.add(row["expected_recipe"])

    metadata = result.get("metadata", {})
    result_ids = {
        result.get("id"),
        metadata.get("id"),
        metadata.get("chunk_id"),
    }
    return bool(expected_ids & result_ids) or metadata.get("recipe_name") in expected_recipes


def _expected_count(row: dict[str, Any]) -> int:
    if row.get("expected_chunk_ids"):
        return len(row["expected_chunk_ids"])
    if row.get("expected_recipes"):
        return len(row["expected_recipes"])
    if row.get("expected_recipe"):
        return 1
    return 1


def _expected_label(row: dict[str, Any]) -> Any:
    return row.get("expected_recipe") or row.get("expected_recipes") or row.get("expected_chunk_ids")


def _snippet(text: str, max_chars: int = 260) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 3].rstrip() + "..."


def _print_search(payload: dict[str, Any]) -> None:
    for i, result in enumerate(payload["results"], start=1):
        metadata = result.get("metadata", {})
        name = metadata.get("recipe_name", "")
        section = metadata.get("section_type", "")
        sources = ",".join(result.get("sources", [result["retriever"]]))
        print(f"{i}. {name} [{section}] score={result['score']:.4f} source={sources}")
        print(f"   id: {result['id']}")
        print(f"   keyword_rank={result.get('keyword_rank')} vector_rank={result.get('vector_rank')}")
        print(f"   {result.get('snippet', '')}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Recherche hybride par mots-clés et embeddings.")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--mode", choices=("keyword", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument("--compare", action="store_true", help="Évaluer les modes mots-clés, vectoriel et hybride")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--build-vectorstore", action="store_true")
    parser.add_argument("--reset-vectorstore", action="store_true")
    parser.add_argument("--jsonl", type=Path, default=CHUNKS_JSONL)
    args = parser.parse_args()

    if args.build_vectorstore:
        rebuild_vectorstore(
            jsonl_path=args.jsonl,
            persist_directory=args.chroma_dir,
            collection_name=args.collection,
            model_name=args.model,
            cache_dir=args.hf_cache_dir,
            reset=args.reset_vectorstore,
        )
        print(f"Base vectorielle prête : {args.chroma_dir} / {args.collection}")

    if args.eval:
        if args.compare:
            print(
                json.dumps(
                    compare_modes(
                        args.eval,
                        persist_directory=args.chroma_dir,
                        collection_name=args.collection,
                        model_name=args.model,
                        cache_dir=args.hf_cache_dir,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        print(
            json.dumps(
                evaluate(
                    args.eval,
                    mode=args.mode,
                    persist_directory=args.chroma_dir,
                    collection_name=args.collection,
                    model_name=args.model,
                    cache_dir=args.hf_cache_dir,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.query:
        payload = run_search(
            args.query,
            mode=args.mode,
            top_k=args.top_k,
            persist_directory=args.chroma_dir,
            collection_name=args.collection,
            model_name=args.model,
            cache_dir=args.hf_cache_dir,
        )
        _print_search(payload)
        return

    print('Passez --query "..." ou --eval data/keyword_eval_queries.jsonl')


if __name__ == "__main__":
    main()
