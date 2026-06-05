"""Small ChatGPT-style web UI for the recipe RAG project.

Run:

    python web_app.py --host 127.0.0.1 --port 8000

The app intentionally uses only the Python standard library for HTTP serving.
It calls the existing retrieval/generation modules directly instead of going
through a shell command, so UTF-8 text stays as Python Unicode strings inside
the process.
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
from p4_rag_generate import DEFAULT_GEMINI_MODEL, generate_answer


APP_TITLE = "Recipe RAG Chat"


def retrieval_answer(query: str, results: list[dict[str, Any]]) -> str:
    """Build a compact answer when no external LLM generation is requested."""
    if not results:
        return "No matching recipe chunks were found."

    lines = [
        f"I found {len(results)} relevant recipe chunk(s) for: {query}",
        "",
        "Top sources:",
    ]
    for index, result in enumerate(results[:5], start=1):
        metadata = repair_metadata(result.get("metadata", {}))
        recipe = metadata.get("recipe_name", "Unknown recipe")
        section = metadata.get("section_type", "unknown")
        source = ",".join(result.get("sources", [result.get("retriever", "")]))
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
        raise ValueError("Query is required.")

    mode = payload.get("mode") or "hybrid"
    if mode not in {"keyword", "vector", "hybrid"}:
        raise ValueError("Mode must be keyword, vector, or hybrid.")

    top_k = int(payload.get("top_k") or 5)
    top_k = max(1, min(top_k, 20))
    generate = bool(payload.get("generate"))

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
                "error": f"Generation failed; showing retrieval results instead. {exc}",
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
        except Exception as exc:  # Keep the UI useful during local debugging.
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
  <title>Recipe RAG Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1d2329;
      --muted: #6a737d;
      --line: #d9ded8;
      --accent: #276b5f;
      --accent-2: #b94e35;
      --chip: #edf4f1;
      --shadow: 0 18px 46px rgba(31, 37, 41, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .app {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #eef1ed;
      padding: 22px;
    }
    .brand {
      font-size: 20px;
      font-weight: 760;
      letter-spacing: 0;
      margin-bottom: 20px;
    }
    .control {
      display: grid;
      gap: 8px;
      margin-bottom: 18px;
    }
    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    select, input[type="number"], input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px 11px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 11px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      color: var(--ink);
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-top: 20px;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      max-height: 100vh;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 18px 26px;
      background: rgba(255, 255, 255, 0.72);
      backdrop-filter: blur(10px);
    }
    header h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 720;
      letter-spacing: 0;
    }
    header p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .messages {
      overflow: auto;
      padding: 28px;
    }
    .message {
      display: grid;
      grid-template-columns: 38px minmax(0, 820px);
      gap: 14px;
      margin: 0 auto 24px;
      max-width: 960px;
    }
    .avatar {
      width: 38px;
      height: 38px;
      border-radius: 7px;
      display: grid;
      place-items: center;
      font-weight: 800;
      color: #fff;
      background: var(--accent);
    }
    .message.user .avatar { background: var(--accent-2); }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px 18px;
      box-shadow: var(--shadow);
      white-space: pre-wrap;
      line-height: 1.55;
    }
    .message.user .bubble {
      box-shadow: none;
      background: #fff8f4;
    }
    .sources {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }
    .source {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfa;
    }
    .source-title {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .chip {
      border-radius: 999px;
      padding: 4px 8px;
      background: var(--chip);
      color: var(--accent);
      font-size: 12px;
      font-weight: 650;
    }
    .composer {
      border-top: 1px solid var(--line);
      padding: 18px 26px 24px;
      background: rgba(247, 247, 244, 0.92);
    }
    form {
      max-width: 960px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr 48px;
      gap: 10px;
      align-items: end;
    }
    textarea {
      resize: none;
      min-height: 54px;
      max-height: 180px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 15px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      line-height: 1.45;
      box-shadow: var(--shadow);
    }
    button {
      width: 48px;
      height: 48px;
      border: 0;
      border-radius: 8px;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
      font-size: 20px;
      font-weight: 800;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .error {
      color: #9f2d20;
      margin-top: 10px;
      font-size: 13px;
    }
    @media (max-width: 820px) {
      .app { grid-template-columns: 1fr; }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        padding: 16px;
      }
      .brand, .hint { grid-column: 1 / -1; margin: 0; }
      main { max-height: none; min-height: 70vh; }
      .messages { padding: 18px 14px; }
      .message { grid-template-columns: 32px minmax(0, 1fr); }
      .avatar { width: 32px; height: 32px; }
      form { grid-template-columns: 1fr 46px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">Recipe RAG Chat</div>
      <div class="control">
        <label for="mode">Retrieval</label>
        <select id="mode">
          <option value="hybrid" selected>Hybrid</option>
          <option value="keyword">Keyword</option>
          <option value="vector">Vector</option>
        </select>
      </div>
      <div class="control">
        <label for="topK">Top K</label>
        <input id="topK" type="number" min="1" max="20" value="5" />
      </div>
      <label class="toggle">
        <input id="generate" type="checkbox" />
        Generate answer
      </label>
      <p class="hint">
        Default mode returns retrieval-backed answers without calling an external LLM.
        Enable generation only when your API key is configured in <code>.env</code>.
      </p>
    </aside>
    <main>
      <header>
        <h1>Ask your recipe corpus</h1>
        <p>Keyword index + vector search, served through the existing Python modules.</p>
      </header>
      <section id="messages" class="messages">
        <div class="message assistant">
          <div class="avatar">R</div>
          <div class="bubble">Ask about ingredients, preparation steps, or recipe names. Try: quels ingredients pour un risotto aux champignons ?</div>
        </div>
      </section>
      <section class="composer">
        <form id="chatForm">
          <textarea id="query" placeholder="Ask a question..." rows="1"></textarea>
          <button id="send" type="submit" title="Send">↑</button>
        </form>
        <div id="error" class="error"></div>
      </section>
    </main>
  </div>
  <script>
    const form = document.querySelector("#chatForm");
    const query = document.querySelector("#query");
    const send = document.querySelector("#send");
    const messages = document.querySelector("#messages");
    const error = document.querySelector("#error");

    function addMessage(role, text, results = []) {
      const node = document.createElement("div");
      node.className = `message ${role}`;
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = role === "user" ? "U" : "R";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      if (results.length) bubble.appendChild(renderSources(results));
      node.appendChild(avatar);
      node.appendChild(bubble);
      messages.appendChild(node);
      messages.scrollTop = messages.scrollHeight;
    }

    function renderSources(results) {
      const wrap = document.createElement("div");
      wrap.className = "sources";
      results.slice(0, 5).forEach((item, index) => {
        const metadata = item.metadata || {};
        const source = document.createElement("div");
        source.className = "source";
        const title = document.createElement("div");
        title.className = "source-title";
        title.innerHTML = `<span>${index + 1}. ${escapeHtml(metadata.recipe_name || "Unknown recipe")}</span><span>${Number(item.score || 0).toFixed(4)}</span>`;
        const body = document.createElement("div");
        body.textContent = item.snippet || item.text || "";
        const chips = document.createElement("div");
        chips.className = "chips";
        [metadata.section_type, ...(item.sources || []), ...(item.matched_terms || [])].filter(Boolean).slice(0, 8).forEach((value) => {
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

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }

    query.addEventListener("input", () => {
      query.style.height = "auto";
      query.style.height = `${Math.min(query.scrollHeight, 180)}px`;
    });

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
      query.style.height = "auto";
      send.disabled = true;
      addMessage("assistant", "Searching...");
      const loading = messages.lastElementChild;
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: text,
            mode: document.querySelector("#mode").value,
            top_k: Number(document.querySelector("#topK").value || 5),
            generate: document.querySelector("#generate").checked
          })
        });
        const payload = await response.json();
        loading.remove();
        if (!response.ok || payload.error) {
          addMessage("assistant", payload.answer || payload.error || "Request failed.", payload.results || []);
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
    # Existing modules use relative data/chroma paths; make them stable no
    # matter where the command is launched from.
    import os

    os.chdir(project_dir)
    server = ThreadingHTTPServer((args.host, args.port), RagChatHandler)
    print(f"{APP_TITLE} running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
