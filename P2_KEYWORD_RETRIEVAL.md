# P2 Keyword Retrieval

This stage adapts the LO17 TD0-TD6 keyword search pipeline to the recipe RAG
corpus while the vector embedding part is still in progress.

## Data

- Input corpus: `data/chunks.jsonl`
- Generated index: `data/keyword_index.json`
- Local evaluation set: `data/keyword_eval_queries.jsonl`

The document unit is one recipe chunk:

- `name`: `metadata.recipe_name`
- `ingredients`: chunks whose `section_type` is `ingredients`
- `steps`: chunks whose `section_type` is `steps`
- `metadata`: section type, servings, difficulty, source URL
- `all`: all fields together

## TD Mapping

- TD0 boolean search: `AND`, `OR`, `NOT`, implicit `AND`
- TD1 corpus preparation: JSONL chunks are loaded as structured documents
- TD2 anti-dictionary: static French stopwords plus high document-frequency terms
- TD3 lemmatization/indexing: lightweight French stemming and field-specific inverted indexes
- TD4 spell correction: prefix candidates plus Levenshtein fallback
- TD5 query processing: `p2_keyword_retrieval.py` exposes `analyze_query()`
- TD6 search engine: `p2_keyword_retrieval.py` is the main search interface

## Build

```bash
python p2_keyword_retrieval.py --rebuild --build-only
```

## Search

Ranked BM25 search:

```bash
python p2_keyword_retrieval.py --query "risotto champignons parmesan" --top-k 5
```

Boolean search:

```bash
python p2_keyword_retrieval.py --query "champignons AND parmesan NOT asperges" --mode boolean
```

Field-specific search:

```bash
python p2_keyword_retrieval.py --query "citron asperges" --field ingredients
python p2_keyword_retrieval.py --query "titre risotto" --top-k 5
```

TD-style main engine:

```bash
python p2_keyword_retrieval.py --query "foie gras figues brioche" --top-k 5 --explain
```

Query analysis is available from Python:

```python
from p2_keyword_retrieval import analyze_query

print(analyze_query("risoto champignns parmesn"))
```

## Evaluation

```bash
python p2_keyword_retrieval.py --eval data/keyword_eval_queries.jsonl
```

Each JSONL test line can use either expected recipe names or exact chunk IDs:

```json
{"query": "risotto champignons parmesan", "expected_recipe": "Risotto aux champignons sauvages"}
{"query": "citron asperges", "expected_chunk_ids": ["Les-delices-du-Risotto-3-recettes-et-particularites-a-decouvrir-ici#3::ingredients"]}
```

The evaluator reports `Recall@1/3/5/10`, `Precision@1/3/5/10`, per-query top IDs,
and average response time.

## Python Usage

```python
from p2_keyword_retrieval import KeywordSearchEngine, keyword_search

engine = KeywordSearchEngine.load()
results, analysis = engine.search("risoto champignns parmesn", top_k=5)

for result in results:
    print(result.chunk_id, result.score, result.metadata["recipe_name"])

payload = keyword_search("risotto champignons parmesan", top_k=5)
print(payload["results"][0]["id"], payload["results"][0]["retriever"])
```

When embeddings are ready, combine these keyword results with vector results.
A simple first fusion rule:

```text
final_score = 0.6 * vector_score + 0.4 * keyword_score
```

## RAG Generation

`p4_rag_generate.py` retrieves context with `p3_hybrid_retrieval.py` and sends
the selected chunks to Gemini or OpenAI.

Create a local `.env` file or set the environment variable:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Generate with Gemini and hybrid retrieval:

```bash
python p4_rag_generate.py --provider gemini --query "Quels ingrédients faut-il pour un risotto aux champignons ?" --top-k 5
```

Use keyword-only retrieval if the embedding model is unavailable:

```bash
python p4_rag_generate.py --provider gemini --retrieval-mode keyword --query "Quels ingrédients faut-il pour un risotto aux champignons ?"
```

Debug the exact retrieved context:

```bash
python p4_rag_generate.py --query "comment préparer un risotto aux asperges ?" --show-context
```

Use OpenAI instead:

```bash
python p4_rag_generate.py --provider openai --model gpt-4.1-mini --query "Quels ingrédients faut-il pour un risotto aux champignons ?"
```
