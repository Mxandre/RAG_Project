"""Streamlit chat interface for the French recipe RAG project.

Run:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from p3_hybrid_retrieval import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION,
    DEFAULT_HF_CACHE_DIR,
    DEFAULT_MODEL,
    repair_metadata,
    repair_mojibake,
    run_search,
)
from p4_rag_generate import DEFAULT_GEMINI_MODEL, classify_query_intent, generate_answer, non_recipe_response


APP_TITLE = "Assistant RAG de recettes"


def _format_retrieval_answer(query: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return "Aucun extrait de recette correspondant n'a ete trouve."

    lines = [
        f"J'ai trouve {len(results)} extrait(s) pertinent(s) pour : {query}",
        "",
        "Meilleures sources :",
    ]
    for index, result in enumerate(results, start=1):
        metadata = repair_metadata(result.get("metadata", {}))
        recipe = metadata.get("recipe_name", "Recette inconnue")
        section = metadata.get("section_type", "section inconnue")
        sources = ", ".join(result.get("sources", [result.get("retriever", "")]))
        snippet = repair_mojibake(result.get("snippet") or result.get("text", ""))
        snippet = " ".join(snippet.split())
        if len(snippet) > 340:
            snippet = snippet[:337].rstrip() + "..."
        lines.append(f"{index}. {recipe} [{section}] via {sources}")
        lines.append(f"   {snippet}")
    return "\n".join(lines)


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "score": float(result.get("score", 0.0) or 0.0),
        "text": repair_mojibake(result.get("text", "")),
        "snippet": repair_mojibake(result.get("snippet", "")),
        "metadata": repair_metadata(result.get("metadata", {})),
        "matched_terms": result.get("matched_terms", []),
        "sources": result.get("sources", [result.get("retriever", "")]),
        "keyword_rank": result.get("keyword_rank"),
        "vector_rank": result.get("vector_rank"),
        "keyword_score": result.get("keyword_score"),
        "vector_score": result.get("vector_score"),
    }


def _run_rag(query: str, *, mode: str, top_k: int, generate: bool) -> dict[str, Any]:
    if not generate and not classify_query_intent(query).get("is_recipe_related"):
        response = non_recipe_response(query)
        response["mode"] = mode
        return response

    if generate:
        try:
            generated = generate_answer(
                query,
                retrieval_top_k=top_k,
                retrieval_mode=mode,
                llm_model=DEFAULT_GEMINI_MODEL,
                persist_directory=DEFAULT_CHROMA_DIR,
                collection_name=DEFAULT_COLLECTION,
                embedding_model=DEFAULT_MODEL,
                hf_cache_dir=DEFAULT_HF_CACHE_DIR,
            )
        except Exception as exc:
            retrieval = run_search(
                query,
                mode=mode,
                top_k=top_k,
                persist_directory=DEFAULT_CHROMA_DIR,
                collection_name=DEFAULT_COLLECTION,
                model_name=DEFAULT_MODEL,
                cache_dir=DEFAULT_HF_CACHE_DIR,
            )
            results = [_compact_result(item) for item in retrieval.get("results", [])]
            return {
                "query": query,
                "mode": mode,
                "answer": _format_retrieval_answer(query, results),
                "error": (
                    "Gemini n'est pas disponible pour cette requete ; "
                    "affichage des meilleurs resultats de recherche a la place. "
                    f"Detail: {_short_error(exc)}"
                ),
                "analysis": retrieval.get("analysis", {}),
                "results": results,
            }
        retrieval = generated.get("retrieval", {"results": []})
        results = [_compact_result(item) for item in retrieval.get("results", [])]
        return {
            "query": query,
            "mode": mode,
            "answer": generated.get("answer") or _format_retrieval_answer(query, results),
            "error": generated.get("error"),
            "analysis": retrieval.get("analysis", {}),
            "results": results,
        }

    retrieval = run_search(
        query,
        mode=mode,
        top_k=top_k,
        persist_directory=DEFAULT_CHROMA_DIR,
        collection_name=DEFAULT_COLLECTION,
        model_name=DEFAULT_MODEL,
        cache_dir=DEFAULT_HF_CACHE_DIR,
    )
    results = [_compact_result(item) for item in retrieval.get("results", [])]
    return {
        "query": query,
        "mode": mode,
        "answer": _format_retrieval_answer(query, results),
        "analysis": retrieval.get("analysis", {}),
        "results": results,
    }


def _short_error(exc: Exception, *, max_chars: int = 220) -> str:
    message = " ".join(str(exc).split())
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3].rstrip() + "..."


def _html_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _message_html(role: str, text: str) -> str:
    avatar = "V" if role == "user" else "R"
    avatar_class = "avatar user" if role == "user" else "avatar"
    bubble_class = "bubble user" if role == "user" else "bubble"
    message_class = "message user" if role == "user" else "message assistant"
    return (
        f'<div class="{message_class}">'
        f'<div class="{avatar_class}">{avatar}</div>'
        f'<div class="{bubble_class}">{_html_escape(text)}</div>'
        "</div>"
    )


def _source_card_html(index: int, result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    recipe = metadata.get("recipe_name", "Recette inconnue")
    section = metadata.get("section_type", "section inconnue")
    snippet = result.get("snippet") or result.get("text") or ""
    score = float(result.get("score", 0.0) or 0.0)
    values = [section, *(result.get("sources") or []), *(result.get("matched_terms") or [])]
    chips = "".join(f'<span class="chip">{_html_escape(item)}</span>' for item in values if item)
    return (
        '<div class="source-card">'
        '<div class="source-title">'
        f"<span>{index}. {_html_escape(recipe)}</span>"
        f"<span>{score:.4f}</span>"
        "</div>"
        f"<div>{_html_escape(snippet)}</div>"
        f'<div class="chips">{chips}</div>'
        "</div>"
    )


def _source_expander_label(index: int, result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {})
    recipe = metadata.get("recipe_name", "Recette inconnue")
    score = float(result.get("score", 0.0) or 0.0)
    return f"Voir le chunk complet #{index} - {recipe} ({score:.4f})"


def _apply_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page: #f5f3ed;
            --panel: #fffdf8;
            --sidebar: #eef3ee;
            --ink: #232522;
            --muted: #66706b;
            --line: #d8d8cd;
            --accent: #315f55;
            --accent-2: #bc4d3b;
            --accent-soft: #e8f0ec;
            --info: #e8f1ff;
            --shadow: 0 18px 45px rgba(43, 47, 43, 0.10);
        }
        header[data-testid="stHeader"] { display: none; }
        #MainMenu, footer { visibility: hidden; }
        .stApp {
            background: var(--page);
            color: var(--ink);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2.2rem;
            padding-bottom: 7.5rem;
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child {
            min-height: 100vh;
            padding-top: 0;
            padding-bottom: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        [data-testid="stSidebar"] h1 {
            color: var(--ink);
            font-size: 1.5rem;
            line-height: 1.1;
            font-weight: 830;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--muted);
        }
        [data-testid="stSidebar"] label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 760;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stButton button {
            min-height: 46px;
            border-radius: 8px;
            border: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: var(--panel);
            color: var(--ink);
            box-shadow: 0 8px 20px rgba(43, 47, 43, 0.06);
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            color: var(--accent) !important;
            fill: var(--accent) !important;
            opacity: 1 !important;
            width: 1.25rem;
            height: 1.25rem;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] [aria-hidden="true"] {
            color: var(--accent) !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] .stButton button {
            background: transparent;
            color: var(--accent);
            border-color: rgba(49, 95, 85, 0.32);
            font-weight: 720;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background: var(--accent-soft);
            color: var(--accent);
            border-color: var(--accent);
        }
        [data-testid="stSidebar"] .stCheckbox label { color: var(--ink); }
        [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
            background: var(--accent-2);
            border-color: var(--accent-2);
        }
        .rag-header {
            max-width: 980px;
            margin: 0 auto 1.35rem;
            border-bottom: 1px solid var(--line);
            padding: 0 0 1.1rem;
        }
        .rag-header h1 {
            margin: 0;
            color: var(--ink);
            font-size: 1.9rem;
            line-height: 1.12;
            font-weight: 850;
            letter-spacing: 0;
        }
        .rag-header p {
            margin: 0.55rem 0 0;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.48;
        }
        .chat-shell {
            max-width: 980px;
            margin: 0 auto 1.25rem;
        }
        .mode-pill {
            display: inline-flex;
            align-items: center;
            margin-left: 0.35rem;
            margin-bottom: 0.35rem;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 760;
        }
        .message,
        .chat-empty {
            display: grid;
            grid-template-columns: 40px minmax(0, 1fr);
            gap: 13px;
            margin-bottom: 1.05rem;
        }
        .message.user {
            grid-template-columns: minmax(0, 1fr) 40px;
            justify-items: end;
        }
        .message.user .avatar {
            grid-column: 2;
        }
        .message.user .bubble {
            grid-column: 1;
            grid-row: 1;
            max-width: min(760px, 100%);
        }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            font-weight: 850;
            color: #fff;
            background: var(--accent);
        }
        .avatar.user { background: var(--accent-2); }
        .bubble {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 1.05rem 1.2rem;
            box-shadow: var(--shadow);
            line-height: 1.55;
            white-space: pre-wrap;
        }
        .bubble.user {
            box-shadow: none;
            background: #fff8f4;
        }
        .source-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem;
            background: #fbfcfa;
            margin-bottom: 0.65rem;
        }
        .source-title {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.5rem;
        }
        .chip {
            border-radius: 999px;
            padding: 0.22rem 0.5rem;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 650;
        }
        .stAlert {
            max-width: 980px;
            margin: 0 auto;
            border-radius: 8px;
            background: var(--info);
        }
        [data-testid="stExpander"] {
            background: #fff;
            border-radius: 8px;
        }
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottomBlockContainer"] > div,
        [data-testid="stChatInput"],
        div:has(> [data-testid="stChatInput"]) {
            background: var(--page) !important;
        }
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"] {
            border-top: 1px solid var(--line);
            box-shadow: 0 -16px 32px rgba(43, 47, 43, 0.07);
        }
        div[data-testid="stChatInput"] {
            max-width: 980px;
            margin: 0 auto;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] > div {
            background: var(--panel) !important;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 8px !important;
            outline: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] [data-baseweb="textarea"],
        div[data-testid="stChatInput"] [data-baseweb="base-input"] {
            background: #ffffff !important;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] > div:focus-within,
        div[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
        div[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] textarea {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            color: var(--ink) !important;
            caret-color: var(--accent-2) !important;
            min-height: 58px !important;
            height: 58px !important;
            border-radius: 8px !important;
            font-size: 1.08rem !important;
            line-height: 58px !important;
            padding: 0 4.2rem 0 1rem !important;
            text-align: left !important;
            box-shadow: 0 12px 30px rgba(43, 47, 43, 0.10);
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #8b938d !important;
            text-align: left !important;
        }
        div[data-testid="stChatInput"] textarea:focus {
            border-color: var(--accent) !important;
            outline: 0 !important;
            caret-color: var(--accent-2) !important;
            box-shadow: 0 0 0 2px rgba(49, 95, 85, 0.14), 0 12px 30px rgba(43, 47, 43, 0.10) !important;
        }
        div[data-testid="stChatInput"] button {
            background: var(--accent) !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        @media (max-width: 820px) {
            .message,
            .chat-empty {
                grid-template-columns: 32px minmax(0, 1fr);
            }
            .avatar {
                width: 32px;
                height: 32px;
            }
            .mode-pill {
                margin-left: 0;
                margin-right: 0.35rem;
                margin-top: 0.75rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_payload", None)
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("pending_query", "")
    st.session_state.setdefault("pending_options", None)


def main() -> None:
    load_dotenv()
    project_dir = Path(__file__).resolve().parent

    st.set_page_config(page_title=APP_TITLE, page_icon="RAG", layout="wide")
    _apply_style()
    _init_state()

    with st.sidebar:
        st.title(APP_TITLE)
        st.caption("Projet LO17 - RAG culinaire francais")
        st.divider()
        mode = st.selectbox(
            "Mode de recherche",
            options=("hybrid", "keyword", "vector"),
            index=0,
            format_func={"hybrid": "Hybride", "keyword": "Mots-cles", "vector": "Vectorielle"}.get,
        )
        top_k = st.slider("Nombre de resultats", min_value=1, max_value=20, value=5)
        generate = st.checkbox("Generer une reponse avec Gemini", value=True)
        st.divider()
        if st.button("Effacer le resultat", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_payload = None
            st.session_state.last_query = ""
            st.session_state.pending_query = ""
            st.session_state.pending_options = None
        st.caption(f"Corpus: {project_dir / 'data' / 'chunks.jsonl'}")

    st.markdown(
        """
        <div class="rag-header">
          <h1>Interrogez votre corpus de recettes</h1>
          <p>Index de mots-cles et recherche vectorielle servis par les modules Python existants.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

    history = st.session_state.history
    payload = st.session_state.last_payload
    pending_query = st.session_state.pending_query
    if history:
        for turn in history:
            st.markdown(_message_html("user", turn.get("query", "")), unsafe_allow_html=True)
            turn_payload = turn.get("payload", {})
            if turn_payload.get("error"):
                st.warning(turn_payload["error"])
            st.markdown(
                _message_html("assistant", turn_payload.get("answer") or "Aucune reponse."),
                unsafe_allow_html=True,
            )

        latest_payload = history[-1].get("payload", {})
        sources_tab, analysis_tab = st.tabs(["Sources de la derniere reponse", "Analyse technique"])
        with sources_tab:
            results = latest_payload.get("results", [])
            if results:
                for index, result in enumerate(results, start=1):
                    st.markdown(_source_card_html(index, result), unsafe_allow_html=True)
                    full_text = result.get("text") or result.get("snippet") or ""
                    if full_text:
                        with st.expander(_source_expander_label(index, result)):
                            st.write(full_text)
            else:
                st.info("Aucune source retrouvee.")
        with analysis_tab:
            st.json(latest_payload.get("analysis", {}))

    if pending_query:
        st.markdown(_message_html("user", pending_query), unsafe_allow_html=True)
        st.markdown(_message_html("assistant", "Recherche en cours..."), unsafe_allow_html=True)
    elif payload and not history:
        if payload.get("error"):
            st.warning(payload["error"])

        st.markdown(_message_html("user", st.session_state.last_query), unsafe_allow_html=True)
        st.markdown(
            _message_html("assistant", payload.get("answer") or "Aucune reponse."),
            unsafe_allow_html=True,
        )

        sources_tab, analysis_tab = st.tabs(["Sources", "Analyse technique"])
        with sources_tab:
            results = payload.get("results", [])
            if results:
                for index, result in enumerate(results, start=1):
                    st.markdown(_source_card_html(index, result), unsafe_allow_html=True)
                    full_text = result.get("text") or result.get("snippet") or ""
                    if full_text:
                        with st.expander(_source_expander_label(index, result)):
                            st.write(full_text)
            else:
                st.info("Aucune source retrouvee.")
        with analysis_tab:
            st.json(payload.get("analysis", {}))
    else:
        st.markdown(
            """
            <div class="chat-empty">
              <div class="avatar">R</div>
              <div class="bubble">Bonjour, je suis pret a chercher dans le corpus. Ecrivez votre question dans la barre de discussion en bas de la page.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input(
        "Posez une question sur les ingredients, les etapes ou les recettes...",
        disabled=bool(st.session_state.pending_query),
    )
    if prompt:
        active_query = prompt.strip()
        if not active_query:
            st.warning("Veuillez saisir une question.")
        else:
            st.session_state.last_payload = None
            st.session_state.last_query = active_query
            st.session_state.pending_query = active_query
            st.session_state.pending_options = {
                "mode": mode,
                "top_k": top_k,
                "generate": generate,
            }
            st.rerun()

    if st.session_state.pending_query:
        active_query = st.session_state.pending_query
        options = st.session_state.pending_options or {
            "mode": mode,
            "top_k": top_k,
            "generate": generate,
        }
        try:
            st.session_state.last_payload = _run_rag(
                active_query,
                mode=options["mode"],
                top_k=options["top_k"],
                generate=options["generate"],
            )
        except Exception as exc:
            st.session_state.last_payload = {
                "query": active_query,
                "mode": options["mode"],
                "answer": f"La requete a echoue : {exc}",
                "error": None,
                "analysis": {},
                "results": [],
            }
        finally:
            st.session_state.last_query = active_query
            st.session_state.history.append(
                {
                    "query": active_query,
                    "payload": st.session_state.last_payload,
                }
            )
            st.session_state.pending_query = ""
            st.session_state.pending_options = None
            st.rerun()


if __name__ == "__main__":
    main()
