"""Génération de réponses RAG à partir des résultats de recherche hybride.

Pipeline:

    question utilisateur
      -> p3_hybrid_retrieval.hybrid_search()
      -> contexte construit à partir des extraits retrouvés
      -> réponse du modèle Gemini

Définissez ``GEMINI_API_KEY`` dans l'environnement ou dans un fichier local
``.env`` avant de lancer la génération. Le mode recherche seule reste
disponible via
``p3_hybrid_retrieval.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from p3_hybrid_retrieval import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, DEFAULT_HF_CACHE_DIR, DEFAULT_MODEL
from p3_hybrid_retrieval import repair_metadata, repair_mojibake
from p3_hybrid_retrieval import run_search


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


INTENT_CLASSIFICATION_PROMPT = """Tu dois décider si la question de l'utilisateur concerne réellement les recettes, la cuisine, les ingrédients, les plats, la nutrition ou la préparation culinaire.

Réponds uniquement avec un objet JSON valide, sans Markdown :
{
  "is_recipe_related": true ou false,
  "intent": "courte catégorie en français",
  "reason": "justification brève en français"
}

Considère comme pertinent :
- une demande de recette, de plat, d'ingrédients, d'étapes, de cuisson ou de substitution ;
- une question nutritionnelle liée à un repas ou à des aliments ;
- une demande de suggestion culinaire à partir d'ingrédients disponibles ;
- une question sur une recette ou un extrait de recette.

Considère comme non pertinent :
- les salutations seules, par exemple "hello", "hi", "bonjour", "nihao" ;
- les tests sans contenu culinaire ;
- les questions qui ne concernent pas la cuisine, les recettes, les aliments ou la nutrition.

En cas de doute, choisis false et laisse l'assistant demander une précision au lieu de lancer une recherche RAG."""


SYSTEM_INSTRUCTIONS = """Tu es un assistant culinaire francophone intégré à un système RAG.

Règles de langue :
- Réponds toujours en français, quelle que soit la langue de la question.
- Si la question contient des termes techniques, des noms de recettes ou des ingrédients dans une autre langue, conserve ces termes si leur traduction risque de créer une ambiguïté.
- Utilise un français clair, naturel et précis, adapté à une interface de recherche de recettes.

Règles de fiabilité :
- Réponds uniquement à partir du contexte fourni par le système RAG.
- Si le contexte ne contient pas l'information nécessaire, dis clairement que tu ne sais pas avec les sources disponibles.
- Ne fabrique jamais d'ingrédients, d'étapes, de quantités, de durées, de conseils ou de sources.
- Ne présente pas une hypothèse comme un fait ; indique explicitement les incertitudes.

Règles de réponse :
- Commence par une réponse directe et utile.
- Donne ensuite les détails pertinents sous forme de paragraphes courts ou de liste si cela améliore la lisibilité.
- Pour une question sur une recette, cite les ingrédients, étapes, temps ou conseils uniquement s'ils apparaissent dans le contexte.
- Mentionne les recettes utilisées et ajoute à la fin une courte section "Sources".
- N'inclus pas de texte en anglais dans les titres ou messages de structure, sauf si le terme vient du contexte ou du nom propre d'une source."""


NON_RECIPE_ANSWER = (
    "Bonjour ! Je peux vous aider à chercher des recettes, des idées de plats, "
    "des ingrédients, des étapes de préparation ou des conseils culinaires. "
    "Posez-moi une question liée à la cuisine pour lancer la recherche."
)

RECIPE_KEYWORDS = {
    "aliment", "aliments", "asiatique", "boeuf", "bœuf", "brocoli", "carotte", "carottes",
    "chou", "cuire", "cuisson", "cuisine", "cuisiner", "courgette", "courgettes", "dessert",
    "diner", "dîner", "etape", "etapes", "étape", "étapes", "farine", "four", "fruit",
    "haricot", "haricots", "ingredient", "ingredients", "ingrédient", "ingrédients",
    "legume", "legumes", "légume", "légumes", "nutrition", "nutritif", "oeuf", "œuf",
    "plat", "plats", "poisson", "porc", "poulet", "preparation", "préparation",
    "recette", "recettes", "repas", "riz", "sauce", "viande",
    "bean", "beans", "beef", "broccoli", "carrot", "chicken", "cook", "cooking",
    "dish", "food", "ingredient", "ingredients", "meal", "meat", "recipe", "vegetable",
    "vegetables", "zucchini",
    "白菜", "胡萝卜", "西兰花", "西葫芦", "豆角", "菜", "菜谱", "营养", "食材", "肉菜",
}

SMALL_TALK_PATTERNS = (
    r"^\s*(hello|hi|hey|bonjour|bonsoir|salut|coucou|ni\s*hao|nihao|你好|您好)\s*[!.。！]*\s*$",
    r"^\s*(merci|thanks|thank you|谢谢)\s*[!.。！]*\s*$",
    r"^\s*(test|测试)\s*[!.。！]*\s*$",
)


def classify_query_intent(query: str, *, llm_model: str | None = None, use_llm: bool = False) -> dict[str, Any]:
    """Détermine si une question doit déclencher une recherche de recettes."""
    heuristic = _classify_query_intent_locally(query)
    if not use_llm or not os.getenv("GEMINI_API_KEY"):
        return heuristic
    try:
        llm_result = _classify_query_intent_with_gemini(query, model=llm_model or DEFAULT_GEMINI_MODEL)
    except Exception:
        return heuristic
    if isinstance(llm_result.get("is_recipe_related"), bool):
        return llm_result
    return heuristic


def is_recipe_related_query(query: str) -> bool:
    return bool(classify_query_intent(query).get("is_recipe_related"))


def non_recipe_response(query: str) -> dict[str, Any]:
    intent = classify_query_intent(query)
    return {
        "query": query,
        "answer": NON_RECIPE_ANSWER,
        "context": "",
        "retrieval": {
            "query": query,
            "analysis": {"intent": intent, "skipped_retrieval": True},
            "results": [],
        },
    }


def _classify_query_intent_locally(query: str) -> dict[str, Any]:
    compact = " ".join(query.lower().strip().split())
    if not compact:
        return {"is_recipe_related": False, "intent": "question vide", "reason": "Aucun contenu à analyser."}
    if any(re.match(pattern, compact, flags=re.I) for pattern in SMALL_TALK_PATTERNS):
        return {"is_recipe_related": False, "intent": "salutation ou test", "reason": "La question ne contient pas de demande culinaire."}
    if any(keyword in compact for keyword in RECIPE_KEYWORDS):
        return {"is_recipe_related": True, "intent": "demande culinaire", "reason": "La question contient des aliments, une recette ou une intention de cuisine."}
    return {"is_recipe_related": False, "intent": "hors sujet culinaire", "reason": "Aucun indice clair de recette, d'ingrédient ou de cuisine."}


def _classify_query_intent_with_gemini(query: str, *, model: str) -> dict[str, Any]:
    response = _generate_with_gemini(
        prompt=f"{INTENT_CLASSIFICATION_PROMPT}\n\nQuestion utilisateur :\n{query}\n\nJSON :",
        model=model,
        temperature=0.0,
        system_instruction="Tu es un classificateur d'intention strict. Réponds uniquement en JSON valide.",
    )
    raw = response.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    return json.loads(raw)


def build_context(results: list[dict[str, Any]], *, max_chars: int = 7000) -> str:
    """Convertit les extraits retrouvés en contexte compact avec sources."""
    blocks: list[str] = []
    used_chars = 0
    for i, result in enumerate(results, start=1):
        metadata = repair_metadata(result.get("metadata", {}))
        text = repair_mojibake(result.get("text", ""))
        block = (
            f"[Source {i}]\n"
            f"Recette : {metadata.get('recipe_name', 'inconnue')}\n"
            f"Section : {metadata.get('section_type', 'inconnue')}\n"
            f"ID extrait : {result.get('id')}\n"
            f"URL: {metadata.get('source_url', '')}\n"
            f"Moteur : {','.join(result.get('sources', [result.get('retriever', '')]))}\n"
            f"Score: {result.get('score', 0):.4f}\n"
            f"Contenu :\n{text}"
        )
        if used_chars + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used_chars += len(block)
    return "\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    return (
        "Instruction prioritaire : réponds intégralement en français.\n\n"
        "Contexte récupéré:\n"
        f"{context}\n\n"
        "Question utilisateur:\n"
        f"{query}\n\n"
        "Réponse en français:"
    )


def generate_answer(
    query: str,
    *,
    retrieval_top_k: int = 5,
    retrieval_mode: str = "hybrid",
    keyword_k: int = 20,
    vector_k: int = 20,
    llm_model: str = DEFAULT_GEMINI_MODEL,
    context_max_chars: int = 7000,
    temperature: float = 0.2,
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_MODEL,
    hf_cache_dir: Path = DEFAULT_HF_CACHE_DIR,
) -> dict[str, Any]:
    """Récupère le contexte puis génère une réponse avec Gemini."""
    load_dotenv()

    intent = classify_query_intent(query, llm_model=llm_model, use_llm=True)
    if not intent.get("is_recipe_related"):
        payload = non_recipe_response(query)
        payload["retrieval"]["analysis"]["intent"] = intent
        return payload

    retrieval = run_search(
        query,
        mode=retrieval_mode,
        top_k=retrieval_top_k,
        persist_directory=persist_directory,
        collection_name=collection_name,
        model_name=embedding_model,
        cache_dir=hf_cache_dir,
        keyword_k=keyword_k,
        vector_k=vector_k,
    )
    context = build_context(retrieval["results"], max_chars=context_max_chars)

    if not os.getenv("GEMINI_API_KEY"):
        return {
            "query": query,
            "answer": None,
            "error": "GEMINI_API_KEY n'est pas défini. Ajoutez-le à l'environnement ou à un fichier .env local.",
            "context": context,
            "retrieval": retrieval,
        }

    answer = _generate_with_gemini(
        prompt=build_prompt(query, context),
        model=llm_model,
        temperature=temperature,
        system_instruction=SYSTEM_INSTRUCTIONS,
    )

    return {
        "query": query,
        "answer": answer,
        "model": llm_model,
        "provider": "gemini",
        "context": context,
        "retrieval": retrieval,
    }


def _generate_with_gemini(
    *,
    prompt: str,
    model: str,
    temperature: float,
    system_instruction: str,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )
    return response.text or ""


def _print_sources(results: list[dict[str, Any]]) -> None:
    print("\nSources retrouvées :")
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

    parser = argparse.ArgumentParser(description="Générer une réponse RAG à partir de la recherche hybride.")
    parser.add_argument("--query", required=True, help="Question utilisateur")
    parser.add_argument("--model", default=None, help="Modèle de génération")
    parser.add_argument("--retrieval-mode", choices=("keyword", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre d'extraits fusionnés utilisés comme contexte")
    parser.add_argument("--keyword-k", type=int, default=20)
    parser.add_argument("--vector-k", type=int, default=20)
    parser.add_argument("--context-max-chars", type=int, default=7000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--json", action="store_true", help="Afficher le JSON complet")
    parser.add_argument("--show-context", action="store_true", help="Afficher le contexte retrouvé")
    args = parser.parse_args()

    payload = generate_answer(
        args.query,
        retrieval_top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        keyword_k=args.keyword_k,
        vector_k=args.vector_k,
        llm_model=args.model or DEFAULT_GEMINI_MODEL,
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
        print("\nLa recherche a réussi ; la génération a été ignorée.")
    else:
        print(payload["answer"])

    _print_sources(payload["retrieval"]["results"])
    if args.show_context:
        print("\nContexte :")
        print(payload["context"])


if __name__ == "__main__":
    main()
