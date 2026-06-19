"""Petite interface web de type chat pour le projet RAG de recettes.

Lancement :

    python web_app.py --host 127.0.0.1 --port 8000

L'application utilise seulement la bibliothèque standard pour servir HTTP. Elle
appelle directement les modules de recherche et de génération afin de garder le
texte UTF-8 sous forme de chaînes Python normales.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from p3_hybrid_retrieval import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, DEFAULT_HF_CACHE_DIR, DEFAULT_MODEL
from p3_hybrid_retrieval import repair_metadata, repair_mojibake, run_search
from p4_rag_generate import DEFAULT_GEMINI_MODEL, classify_query_intent, generate_answer, non_recipe_response


APP_TITLE = "Recipe RAG assistant"


def retrieval_answer(query: str, results: list[dict[str, Any]]) -> str:
    """Construit une réponse courte sans appeler de LLM externe."""
    if not results:
        return "No matching recipe chunks were found."

    lines = [
        f"Found {len(results)} relevant chunk(s) for: {query}",
        "",
        "Top sources:",
    ]
    for index, result in enumerate(results, start=1):
        metadata = repair_metadata(result.get("metadata", {}))
        recipe = metadata.get("recipe_name", "Unknown recipe")
        section = metadata.get("section_type", "unknown")
        source = ", ".join(result.get("sources", [result.get("retriever", "")]))
        snippet = repair_mojibake(result.get("snippet") or result.get("text", ""))
        snippet = " ".join(snippet.split())
        if len(snippet) > 340:
            snippet = snippet[:337].rstrip() + "..."
        lines.append(f"{index}. {recipe} [{section}] via {source}")
        lines.append(f"   {snippet}")
    return "\n".join(lines)


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = repair_metadata(result.get("metadata", {}))
    text = repair_mojibake(result.get("text", ""))
    snippet = repair_mojibake(result.get("snippet", ""))
    return {
        "id": result.get("id"),
        "score": result.get("score", 0.0),
        "text": text,
        "snippet": snippet,
        "metadata": metadata,
        "matched_terms": result.get("matched_terms", []),
        "sources": result.get("sources", [result.get("retriever", "")]),
        "keyword_rank": result.get("keyword_rank"),
        "vector_rank": result.get("vector_rank"),
        "keyword_score": result.get("keyword_score"),
        "vector_score": result.get("vector_score"),
    }


def handle_chat(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("A question is required.")

    mode = payload.get("mode") or "hybrid"
    if mode not in {"keyword", "vector", "hybrid"}:
        raise ValueError("Mode must be keyword, vector, or hybrid.")

    top_k = int(payload.get("top_k") or 6)
    top_k = max(1, min(top_k, 20))
    generate = bool(payload.get("generate"))

    if not generate and not classify_query_intent(query).get("is_recipe_related"):
        response = non_recipe_response(query)
        response["mode"] = mode
        return response

    if generate:
        model = payload.get("model") or DEFAULT_GEMINI_MODEL
        try:
            generated = generate_answer(
                query,
                retrieval_top_k=top_k,
                retrieval_mode=mode,
                llm_model=model,
                persist_directory=DEFAULT_CHROMA_DIR,
                collection_name=DEFAULT_COLLECTION,
                embedding_model=DEFAULT_MODEL,
                hf_cache_dir=DEFAULT_HF_CACHE_DIR,
            )
        except Exception as exc:
            if not classify_query_intent(query).get("is_recipe_related"):
                response = non_recipe_response(query)
                response["mode"] = mode
                return response
            retrieval = run_search(
                query,
                mode=mode,
                top_k=top_k,
                persist_directory=DEFAULT_CHROMA_DIR,
                collection_name=DEFAULT_COLLECTION,
                model_name=DEFAULT_MODEL,
                cache_dir=DEFAULT_HF_CACHE_DIR,
            )
            results = [compact_result(item) for item in retrieval.get("results", [])]
            return {
                "query": query,
                "mode": mode,
                "answer": retrieval_answer(query, results),
                "error": f"Generation failed. Showing retrieval results instead. {exc}",
                "analysis": retrieval.get("analysis", {}),
                "results": results,
            }
        retrieval = generated["retrieval"]
        results = [compact_result(item) for item in retrieval.get("results", [])]
        return {
            "query": query,
            "mode": mode,
            "answer": generated.get("answer") or retrieval_answer(query, results),
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
    results = [compact_result(item) for item in retrieval.get("results", [])]
    return {
        "query": query,
        "mode": mode,
        "answer": retrieval_answer(query, results),
        "analysis": retrieval.get("analysis", {}),
        "results": results,
    }


class RagChatHandler(BaseHTTPRequestHandler):
    server_version = "RecipeRAGChat/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if path == "/health":
            self._send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            payload = self._read_json()
            self._send_json(handle_chat(payload))
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Recipe RAG assistant</title>
  <style>
    :root {
      color-scheme: light;
      --canvas: #f4f1ea;
      --paper: #fffdf8;
      --paper-2: #f9f7f0;
      --ink: #22211f;
      --soft-ink: #5f625f;
      --muted: #8a8a82;
      --line: #ded8ca;
      --line-strong: #c8bfad;
      --sage: #426f62;
      --sage-strong: #2d5148;
      --tomato: #b85439;
      --saffron: #d9a441;
      --aubergine: #463249;
      --blue: #3d5d8f;
      --shadow: 0 18px 50px rgba(55, 48, 38, 0.13);
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      overflow: hidden;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.48) 1px, transparent 1px) 0 0 / 32px 32px,
        linear-gradient(180deg, rgba(255,255,255,0.38) 1px, transparent 1px) 0 0 / 32px 32px,
        var(--canvas);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, textarea { font: inherit; }
    button { cursor: pointer; }
    .app {
      display: grid;
      grid-template-columns: 292px minmax(420px, 1fr) 326px;
      height: 100vh;
      min-width: 0;
    }
    .rail {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 22px;
      border-right: 1px solid var(--line);
      background: rgba(255, 253, 248, 0.76);
      padding: 26px 22px 24px;
      backdrop-filter: blur(16px);
    }
    .mark {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .mark-icon {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: var(--ink);
      color: var(--paper);
      font-weight: 850;
      letter-spacing: 0;
    }
    .mark-title {
      margin: 0;
      font-size: 18px;
      line-height: 1.12;
      font-weight: 820;
      letter-spacing: 0;
    }
    .mark-kicker {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .controls {
      display: grid;
      gap: 22px;
      align-content: start;
    }
    .field {
      display: grid;
      gap: 10px;
    }
    .field-label {
      color: var(--soft-ink);
      font-size: 12px;
      font-weight: 780;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .segments {
      display: grid;
      grid-template-columns: 1fr;
      gap: 7px;
    }
    .segment {
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      color: var(--ink);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
      font-weight: 760;
    }
    .segment span {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--line-strong);
    }
    .segment.active {
      color: var(--paper);
      background: var(--sage-strong);
      border-color: var(--sage-strong);
      box-shadow: 0 10px 22px rgba(45, 81, 72, 0.22);
    }
    .segment.active span { background: var(--saffron); }
    .number-row {
      display: grid;
      grid-template-columns: 42px 1fr 42px;
      gap: 8px;
      align-items: center;
    }
    .step, .send {
      border: 0;
      border-radius: 8px;
      background: var(--ink);
      color: var(--paper);
    }
    .step {
      width: 42px;
      height: 42px;
      font-size: 20px;
    }
    input[type="number"] {
      width: 100%;
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      color: var(--ink);
      text-align: center;
      font-size: 18px;
      font-weight: 760;
      outline: none;
    }
    .toggle {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 12px;
      align-items: start;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
    }
    .toggle input {
      width: 18px;
      height: 18px;
      margin: 2px 0 0;
      accent-color: var(--sage);
    }
    .toggle strong {
      display: block;
      font-size: 14px;
      line-height: 1.2;
    }
    .toggle small {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      line-height: 1.35;
    }
    .note {
      border-left: 3px solid var(--saffron);
      padding-left: 12px;
      color: var(--soft-ink);
      font-size: 13px;
      line-height: 1.5;
    }
    .stage {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      height: 100vh;
    }
    .topbar {
      border-bottom: 1px solid var(--line);
      background: rgba(244, 241, 234, 0.82);
      backdrop-filter: blur(16px);
      padding: 22px 34px 18px;
    }
    .eyebrow {
      color: var(--tomato);
      font-size: 12px;
      font-weight: 830;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .topbar h1 {
      margin: 6px 0 0;
      max-width: 780px;
      font-size: 31px;
      line-height: 1.05;
      font-weight: 860;
      letter-spacing: 0;
    }
    .topbar p {
      margin: 9px 0 0;
      max-width: 720px;
      color: var(--soft-ink);
      font-size: 15px;
      line-height: 1.45;
    }
    .messages {
      overflow: auto;
      padding: 30px 34px;
      scroll-behavior: smooth;
    }
    .message {
      display: grid;
      grid-template-columns: 42px minmax(0, 760px);
      gap: 14px;
      margin-bottom: 22px;
      align-items: start;
    }
    .message.user {
      grid-template-columns: minmax(0, 760px) 42px;
      justify-content: end;
    }
    .avatar {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: var(--paper);
      background: var(--sage);
      font-weight: 850;
    }
    .message.user .avatar {
      grid-column: 2;
      background: var(--tomato);
    }
    .message.user .bubble { grid-row: 1; }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 253, 248, 0.94);
      box-shadow: var(--shadow);
      padding: 18px 20px;
      white-space: pre-wrap;
      line-height: 1.58;
      font-size: 16px;
    }
    .message.user .bubble {
      background: #fff7ef;
      box-shadow: none;
      border-color: #ead2c0;
    }
    .welcome {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
      padding: 20px;
    }
    .welcome h2 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }
    .welcome p {
      margin: 8px 0 0;
      color: var(--soft-ink);
      line-height: 1.48;
    }
    .prompts {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
    }
    .prompt {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--paper-2);
      color: var(--ink);
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 720;
    }
    .sources {
      display: grid;
      gap: 9px;
      margin-top: 16px;
      white-space: normal;
    }
    .source {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper-2);
      padding: 12px;
      font-size: 13px;
    }
    .source-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
      font-weight: 780;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 9px;
    }
    .chip {
      border-radius: 999px;
      background: #e9efe8;
      color: var(--sage-strong);
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 760;
    }
    .composer {
      border-top: 1px solid var(--line);
      background: rgba(244, 241, 234, 0.88);
      backdrop-filter: blur(16px);
      padding: 18px 34px 24px;
    }
    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 54px;
      gap: 12px;
      max-width: 860px;
      margin: 0 auto;
      align-items: end;
    }
    textarea {
      width: 100%;
      min-height: 58px;
      max-height: 170px;
      resize: none;
      border: 2px solid var(--ink);
      border-radius: 8px;
      background: var(--paper);
      color: var(--ink);
      padding: 16px 17px;
      font-size: 16px;
      line-height: 1.42;
      outline: none;
      box-shadow: 0 10px 26px rgba(55, 48, 38, 0.10);
    }
    textarea:focus {
      border-color: var(--sage);
      box-shadow: 0 0 0 3px rgba(66, 111, 98, 0.18);
    }
    .send {
      width: 54px;
      height: 54px;
      font-size: 23px;
      background: var(--sage);
    }
    .send:hover { background: var(--sage-strong); }
    .send:disabled {
      cursor: wait;
      opacity: 0.62;
    }
    .error {
      max-width: 860px;
      margin: 10px auto 0;
      color: #9f2d20;
      font-size: 13px;
    }
    .inspector {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 18px;
      border-left: 1px solid var(--line);
      background: rgba(35, 33, 31, 0.95);
      color: var(--paper);
      padding: 26px 22px;
      min-width: 0;
    }
    .inspector h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    .inspector p {
      margin: 7px 0 0;
      color: #c7c1b6;
      font-size: 13px;
      line-height: 1.42;
    }
    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 9px;
      margin-top: 16px;
    }
    .stat {
      border: 1px solid rgba(255,255,255,0.13);
      border-radius: 8px;
      padding: 11px;
      background: rgba(255,255,255,0.06);
    }
    .stat small {
      display: block;
      color: #b9b2a6;
      font-size: 11px;
      font-weight: 760;
      text-transform: uppercase;
    }
    .stat strong {
      display: block;
      margin-top: 6px;
      font-size: 17px;
    }
    .source-list {
      min-height: 0;
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 10px;
      padding-right: 4px;
    }
    .mini-source {
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 8px;
      background: rgba(255,255,255,0.06);
      padding: 12px;
    }
    .mini-source strong {
      display: block;
      font-size: 13px;
      line-height: 1.3;
    }
    .mini-source p {
      margin: 8px 0 0;
      color: #d9d2c6;
      font-size: 12px;
    }
    .empty-state {
      border: 1px dashed rgba(255,255,255,0.20);
      border-radius: 8px;
      color: #c7c1b6;
      padding: 14px;
      font-size: 13px;
      line-height: 1.45;
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 272px minmax(0, 1fr); }
      .inspector { display: none; }
    }
    @media (max-width: 760px) {
      body { overflow: auto; }
      .app { display: block; height: auto; min-height: 100vh; }
      .rail {
        display: block;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .controls {
        margin-top: 18px;
        grid-template-columns: 1fr;
      }
      .stage { height: auto; min-height: 70vh; }
      .topbar, .messages, .composer { padding-left: 18px; padding-right: 18px; }
      .topbar h1 { font-size: 25px; }
      .message, .message.user {
        grid-template-columns: 36px minmax(0, 1fr);
        justify-content: stretch;
      }
      .message.user .avatar { grid-column: 1; }
      .message.user .bubble { grid-column: 2; grid-row: 1; }
      .avatar { width: 36px; height: 36px; }
      form { grid-template-columns: 1fr 50px; }
      .send { width: 50px; height: 50px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="rail">
      <div class="mark">
        <div class="mark-icon">R</div>
        <div>
          <h1 class="mark-title">Recipe RAG<br />assistant</h1>
          <div class="mark-kicker">Recipe search</div>
        </div>
      </div>

      <div class="controls">
        <div class="field">
          <div class="field-label">Search engine</div>
          <input id="mode" type="hidden" value="hybrid" />
          <div class="segments" role="group" aria-label="Search mode">
            <button class="segment active" type="button" data-mode="hybrid">Hybrid <span></span></button>
            <button class="segment" type="button" data-mode="keyword">Keyword <span></span></button>
            <button class="segment" type="button" data-mode="vector">Vector <span></span></button>
          </div>
        </div>

        <div class="field">
          <label class="field-label" for="topK">Number of results</label>
          <div class="number-row">
            <button class="step" type="button" data-step="-1" aria-label="Decrease">-</button>
            <input id="topK" type="number" min="1" max="20" value="6" />
            <button class="step" type="button" data-step="1" aria-label="Increase">+</button>
          </div>
        </div>

        <label class="toggle">
          <input id="generate" type="checkbox" checked />
          <span>
            <strong>Generated answer</strong>
            <small>Uses the LLM when the API key is configured. Otherwise, it shows the best passages.</small>
          </span>
        </label>
      </div>

      <div class="note">
        The existing Python modules still do the work. This page only changes how the results are shown.
      </div>
    </aside>

    <main class="stage">
      <header class="topbar">
        <div class="eyebrow">Recipe corpus / LO17</div>
        <h1>Search ingredients, steps, and source chunks from one page.</h1>
        <p>Ask a normal question. The system can use keyword search, vector search, and optional generation.</p>
      </header>

      <section id="messages" class="messages">
        <div class="message assistant">
          <div class="avatar">R</div>
          <div class="welcome">
            <h2>What do you want to cook?</h2>
            <p>Ask for a recipe, ingredients, preparation steps, or an idea based on what you have.</p>
            <div class="prompts">
              <button class="prompt" type="button">What ingredients do I need for mushroom risotto?</button>
              <button class="prompt" type="button">I want to make ratatouille.</button>
              <button class="prompt" type="button">Suggest a recipe with chicken and vegetables.</button>
            </div>
          </div>
        </div>
      </section>

      <section class="composer">
        <form id="chatForm">
          <textarea id="query" placeholder="Ask a question..." rows="1"></textarea>
          <button id="send" class="send" type="submit" title="Send" aria-label="Send">&uarr;</button>
        </form>
        <div id="error" class="error"></div>
      </section>
    </main>

    <aside class="inspector" aria-label="Search details">
      <div>
        <h2>Search details</h2>
        <p id="inspectorSummary">No query has run yet.</p>
        <div class="status-grid">
          <div class="stat"><small>Mode</small><strong id="statMode">Hybrid</strong></div>
          <div class="stat"><small>Top K</small><strong id="statTopK">6</strong></div>
        </div>
      </div>
      <div id="sourceList" class="source-list">
        <div class="empty-state">Retrieved sources will appear here after the first question.</div>
      </div>
    </aside>
  </div>

  <script>
    const form = document.querySelector("#chatForm");
    const query = document.querySelector("#query");
    const send = document.querySelector("#send");
    const messages = document.querySelector("#messages");
    const error = document.querySelector("#error");
    const modeInput = document.querySelector("#mode");
    const topKInput = document.querySelector("#topK");
    const sourceList = document.querySelector("#sourceList");
    const inspectorSummary = document.querySelector("#inspectorSummary");
    const statMode = document.querySelector("#statMode");
    const statTopK = document.querySelector("#statTopK");
    const modeLabels = { hybrid: "Hybrid", keyword: "Keyword", vector: "Vector" };

    document.querySelectorAll(".segment").forEach((button) => {
      button.addEventListener("click", () => {
        modeInput.value = button.dataset.mode;
        document.querySelectorAll(".segment").forEach((item) => item.classList.toggle("active", item === button));
        syncInspectorShell();
      });
    });

    document.querySelectorAll("[data-step]").forEach((button) => {
      button.addEventListener("click", () => {
        const next = Number(topKInput.value || 6) + Number(button.dataset.step);
        topKInput.value = Math.max(1, Math.min(20, next));
        syncInspectorShell();
      });
    });

    topKInput.addEventListener("input", syncInspectorShell);

    document.querySelectorAll(".prompt").forEach((button) => {
      button.addEventListener("click", () => {
        query.value = button.textContent.trim();
        resizeQuery();
        query.focus();
      });
    });

    function syncInspectorShell() {
      statMode.textContent = modeLabels[modeInput.value] || modeInput.value;
      statTopK.textContent = String(topKInput.value || 6);
    }

    function addMessage(role, text, results = []) {
      const node = document.createElement("div");
      node.className = `message ${role}`;
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = role === "user" ? "V" : "R";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      if (results.length) bubble.appendChild(renderSources(results));

      if (role === "user") {
        node.appendChild(bubble);
        node.appendChild(avatar);
      } else {
        node.appendChild(avatar);
        node.appendChild(bubble);
      }
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
    }

    function renderSources(results) {
      const wrap = document.createElement("div");
      wrap.className = "sources";
      results.slice(0, 3).forEach((item, index) => {
        const metadata = item.metadata || {};
        const source = document.createElement("div");
        source.className = "source";
        const title = document.createElement("div");
        title.className = "source-title";
        title.innerHTML = `<span>${index + 1}. ${escapeHtml(metadata.recipe_name || "Unknown recipe")}</span><span>${Number(item.score || 0).toFixed(3)}</span>`;
        const body = document.createElement("div");
        body.textContent = item.snippet || item.text || "";
        const chips = document.createElement("div");
        chips.className = "chips";
        [metadata.section_type, ...(item.sources || []), ...(item.matched_terms || [])].filter(Boolean).slice(0, 7).forEach((value) => {
          const chip = document.createElement("span");
          chip.className = "chip";
          chip.textContent = value;
          chips.appendChild(chip);
        });
        source.appendChild(title);
        source.appendChild(body);
        source.appendChild(chips);
        wrap.appendChild(source);
      });
      return wrap;
    }

    function updateInspector(payload) {
      const results = payload.results || [];
      inspectorSummary.textContent = results.length
        ? `${results.length} source(s) found for "${payload.query || ""}".`
        : `No displayable source for "${payload.query || ""}".`;
      sourceList.innerHTML = "";
      if (!results.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "The answer does not include a source passage yet.";
        sourceList.appendChild(empty);
        return;
      }
      results.forEach((item, index) => {
        const metadata = item.metadata || {};
        const card = document.createElement("div");
        card.className = "mini-source";
        const title = document.createElement("strong");
        title.textContent = `${index + 1}. ${metadata.recipe_name || "Unknown recipe"}`;
        const meta = document.createElement("p");
        meta.textContent = `${metadata.section_type || "section"} / score ${Number(item.score || 0).toFixed(3)}`;
        card.appendChild(title);
        card.appendChild(meta);
        sourceList.appendChild(card);
      });
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }

    function resizeQuery() {
      query.style.height = "auto";
      query.style.height = `${Math.min(query.scrollHeight, 170)}px`;
    }

    query.addEventListener("input", resizeQuery);
    query.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = query.value.trim();
      if (!text) return;
      error.textContent = "";
      addMessage("user", text);
      query.value = "";
      resizeQuery();
      send.disabled = true;
      addMessage("assistant", "Searching...");
      const loading = messages.lastElementChild;
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: text,
            mode: modeInput.value,
            top_k: Number(topKInput.value || 6),
            generate: document.querySelector("#generate").checked
          })
        });
        const payload = await response.json();
        loading.remove();
        updateInspector(payload);
        if (!response.ok || payload.error) {
          addMessage("assistant", payload.answer || payload.error || "The query failed.", payload.results || []);
          if (payload.error) error.textContent = payload.error;
        } else {
          addMessage("assistant", payload.answer || "No answer.", payload.results || []);
        }
      } catch (err) {
        loading.remove();
        error.textContent = err.message || String(err);
        addMessage("assistant", "The local server returned an error.");
      } finally {
        send.disabled = false;
        query.focus();
      }
    });

    syncInspectorShell();
  </script>
</body>
</html>
"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=f"Run {APP_TITLE}.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    # Les modules existants utilisent des chemins relatifs vers data et Chroma.
    # On fixe donc le dossier de travail, quel que soit le point de lancement.
    import os

    os.chdir(project_dir)
    server = ThreadingHTTPServer((args.host, args.port), RagChatHandler)
    print(f"{APP_TITLE} available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
