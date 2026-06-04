"""TD-style keyword search engine adapted to the recipe RAG corpus.

The recipe corpus is stored as JSONL chunks:

    data/chunks.jsonl

This module adapts the LO17 TD0-TD6 pipeline to those chunks:

- corpus preparation from JSONL chunks
- tokenization and normalization
- anti-dictionary / stopword filtering
- lightweight French stemming
- field-specific inverted indexes
- reverse dictionary
- spelling correction with prefix candidates + Levenshtein fallback
- natural-language query processing
- Boolean and ranked BM25 search
- local precision/recall evaluation

It is dependency-free on purpose, so it can run before the embedding part of
the RAG system is ready.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


DATA_DIR = Path("data")
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
KEYWORD_INDEX_JSON = DATA_DIR / "keyword_index.json"

INDEX_VERSION = 2
FIELDS = ("all", "name", "ingredients", "steps", "metadata")


STOPWORDS = {
    "a", "afin", "ai", "aient", "ainsi", "ait", "alors", "apres", "au",
    "aucun", "aussi", "autre", "aux", "avaient", "avais", "avait", "avec",
    "avoir", "ce", "cela", "ces", "cet", "cette", "ceux", "chaque",
    "comme", "comment", "dans", "de", "des", "du", "elle", "elles", "en",
    "encore", "est", "et", "etre", "eu", "fait", "faites", "font", "il",
    "ils", "je", "jusqu", "la", "le", "les", "leur", "leurs", "lors",
    "mais", "me", "mes", "ne", "nos", "notre", "nous", "on", "ont", "ou",
    "par", "pas", "pendant", "plus", "pour", "puis", "qu", "que", "quel",
    "quelle", "quelles", "quels", "qui", "quoi", "sa", "se", "selon",
    "ses", "si", "son", "sont", "sur", "ta", "te", "tes", "toi", "ton",
    "tout", "tres", "tu", "un", "une", "vos", "votre", "vous", "y",
    # Query boilerplate.
    "afficher", "chercher", "donner", "je", "liste", "recette", "recettes",
    "voudrais", "veux", "souhaite", "trouve", "trouver",
}


# Known mojibake seen in the current data when French UTF-8 text was decoded
# through a Chinese code page. Using escapes keeps this source file ASCII-safe.
MOJIBAKE_REPLACEMENTS = {
    "\u8305": "e",   # e acute in the scraped artifacts
    "\u732b": "e",   # e grave
    "\u951a": "e",   # e circumflex
    "\u813f": "a",   # a grave
    "\u8292": "a",   # a circumflex
    "\u83bd": "c",   # c cedilla
    "\u536f": "i",   # i circumflex
    "\u4e48": "o",   # o circumflex
    "\u6ca1": "u",   # u circumflex
    "\u8259": "oe",
    "\u9205": " ",
    "\u6aab": " ",
    "\u6416": " ",
    "\u63b3": " ",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9']*")
NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")

EXPLICIT_OPERATORS = {
    "and": "AND",
    "et": "AND",
    "or": "OR",
    "ou": "OR",
    "not": "NOT",
    "non": "NOT",
    "sans": "NOT",
}

FIELD_ALIASES = {
    "ingredient": "ingredients",
    "ingredients": "ingredients",
    "ingr": "ingredients",
    "etape": "steps",
    "etapes": "steps",
    "preparation": "steps",
    "instruction": "steps",
    "instructions": "steps",
    "titre": "name",
    "nom": "name",
    "name": "name",
    "serving": "metadata",
    "servings": "metadata",
    "personne": "metadata",
    "personnes": "metadata",
}


@dataclass
class ChunkDocument:
    id: str
    text: str
    metadata: dict


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    keywords: list[str]
    corrected_keywords: list[str]
    corrections: dict[str, str]
    unknown_terms: list[str]
    operators: list[str]
    field_filter: str | None = None
    boolean_expression: str | None = None
    phrases: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    matched_terms: list[str]
    text: str
    metadata: dict
    snippet: str = ""


def normalize_text(text: str) -> str:
    """Lowercase, deaccent, normalize apostrophes, and repair known artifacts."""
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("\u2019", "'").replace("`", "'")
    return text


def raw_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        token = token.strip("'")
        if "'" in token:
            prefix, rest = token.rsplit("'", 1)
            if prefix in {"d", "l", "j", "m", "n", "s", "t", "qu"} and rest:
                token = rest
        if token:
            tokens.append(token)
    return tokens


def stem_french(token: str) -> str:
    """A compact French stemmer inspired by the TD Snowball step.

    This is not a full linguistic lemmatizer. It is a local, dependency-free
    normalization that groups common plural, gender, and verb variants well
    enough for recipe search.
    """
    if NUMBER_RE.match(token) or len(token) <= 3:
        return token

    suffixes = (
        "issements", "issement", "atrices", "ateurs", "ations", "logies",
        "ements", "ement", "euses", "euse", "eaux", "aux", "iques",
        "ique", "ment", "ments", "ances", "ance", "ences", "ence",
        "ites", "ite", "ees", "ee", "es", "s",
    )
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix in {"s", "es"}:
                return token[: -len(suffix)]
            if suffix == "aux":
                return token[:-3] + "al"
            return token[: -len(suffix)]
    return token


def analyze_tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = raw_tokens(text)
    out: list[str] = []
    for token in tokens:
        if not keep_stopwords and token in STOPWORDS:
            continue
        stem = stem_french(token)
        if keep_stopwords or (len(stem) > 1 and stem not in STOPWORDS):
            out.append(stem)
    return out


def load_chunks(path: Path = CHUNKS_JSONL) -> list[ChunkDocument]:
    with path.open(encoding="utf-8") as f:
        return [ChunkDocument(**json.loads(line)) for line in f if line.strip()]


def _field_text(doc: ChunkDocument, field_name: str) -> str:
    section = doc.metadata.get("section_type", "")
    if field_name == "all":
        return " ".join(
            [
                doc.metadata.get("recipe_name", ""),
                section,
                doc.metadata.get("servings", ""),
                doc.metadata.get("difficulty", ""),
                doc.text,
            ]
        )
    if field_name == "name":
        return doc.metadata.get("recipe_name", "")
    if field_name == "ingredients":
        return doc.text if section == "ingredients" else ""
    if field_name == "steps":
        return doc.text if section == "steps" else ""
    if field_name == "metadata":
        return " ".join(
            [
                doc.metadata.get("section_type", ""),
                doc.metadata.get("servings", ""),
                doc.metadata.get("difficulty", ""),
                doc.metadata.get("source_url", ""),
            ]
        )
    raise ValueError(f"Unknown field: {field_name}")


class KeywordSearchEngine:
    """Complete TD-style keyword engine for recipe chunks."""

    def __init__(self, documents: list[ChunkDocument], *, auto_stopword_df: float = 0.85) -> None:
        self.documents = documents
        self.doc_by_id = {doc.id: doc for doc in documents}
        self.all_doc_ids = set(self.doc_by_id)

        self.anti_dictionary: set[str] = set(STOPWORDS)
        self.lexicon: set[str] = set()
        self.token_to_stem: dict[str, str] = {}

        self.doc_lengths: dict[str, dict[str, int]] = {field_name: {} for field_name in FIELDS}
        self.reverse_dictionary: dict[str, dict[str, dict[str, int]]] = {}
        self.inverted_indexes: dict[str, dict[str, dict[str, int]]] = {
            field_name: defaultdict(dict) for field_name in FIELDS
        }
        self.idf: dict[str, dict[str, float]] = {field_name: {} for field_name in FIELDS}
        self.avg_doc_len: dict[str, float] = {field_name: 0.0 for field_name in FIELDS}
        self.tf: dict[str, dict[str, dict[str, int]]] = {}
        self.tfidf: dict[str, dict[str, dict[str, float]]] = {}

        self._build(auto_stopword_df=auto_stopword_df)

    def _build(self, *, auto_stopword_df: float) -> None:
        self._build_token_maps()
        self._build_anti_dictionary(auto_stopword_df)
        for field_name in FIELDS:
            self._build_field_index(field_name)

    def _build_token_maps(self) -> None:
        for doc in self.documents:
            for token in raw_tokens(_field_text(doc, "all")):
                stem = stem_french(token)
                self.token_to_stem[token] = stem
                self.lexicon.add(stem)

    def _build_anti_dictionary(self, auto_stopword_df: float) -> None:
        doc_freq: Counter[str] = Counter()
        n_docs = max(len(self.documents), 1)
        for doc in self.documents:
            doc_freq.update(set(stem_french(tok) for tok in raw_tokens(_field_text(doc, "all"))))

        for term, df in doc_freq.items():
            if term in STOPWORDS or len(term) <= 1:
                self.anti_dictionary.add(term)
            elif df / n_docs >= auto_stopword_df and not NUMBER_RE.match(term):
                self.anti_dictionary.add(term)

        self.lexicon = {term for term in self.lexicon if term not in self.anti_dictionary}

    def _build_field_index(self, field_name: str) -> None:
        total_len = 0
        self.tf[field_name] = {}
        self.tfidf[field_name] = {}

        for doc in self.documents:
            terms = [
                term
                for term in analyze_tokens(_field_text(doc, field_name))
                if term not in self.anti_dictionary
            ]
            counts = Counter(terms)
            self.reverse_dictionary.setdefault(doc.id, {})[field_name] = dict(counts)
            self.tf[field_name][doc.id] = dict(counts)
            self.doc_lengths[field_name][doc.id] = len(terms)
            total_len += len(terms)
            for term, freq in counts.items():
                self.inverted_indexes[field_name][term][doc.id] = freq

        n_docs = max(len(self.documents), 1)
        self.avg_doc_len[field_name] = total_len / n_docs
        for term, postings in self.inverted_indexes[field_name].items():
            df = len(postings)
            self.idf[field_name][term] = math.log10(n_docs / df) if df else 0.0

        for doc_id, counts in self.tf[field_name].items():
            self.tfidf[field_name][doc_id] = {
                term: freq * self.idf[field_name].get(term, 0.0)
                for term, freq in counts.items()
            }

    def analyze_query(self, query: str) -> QueryAnalysis:
        normalized = normalize_text(query)
        phrases = re.findall(r'"([^"]+)"', normalized)
        working = re.sub(r'"[^"]+"', " ", normalized)
        working = re.sub(r"\bmais\s+pas\b", " NOT ", working)
        working = re.sub(r"\bsans\b", " NOT ", working)

        field_filter = self._extract_field_filter(working)
        operators = []
        keywords = []
        expression_parts = []
        previous_was_term = False

        for token in raw_tokens(working):
            op = EXPLICIT_OPERATORS.get(token)
            if op:
                if op == "NOT" and previous_was_term:
                    expression_parts.append("AND")
                    operators.append("AND")
                operators.append(op)
                expression_parts.append(op)
                previous_was_term = False
                continue
            if token in FIELD_ALIASES or token in self.anti_dictionary:
                continue
            stem = stem_french(token)
            if stem in self.anti_dictionary or len(stem) <= 1:
                continue
            if previous_was_term:
                expression_parts.append("AND")
                operators.append("AND")
            keywords.append(stem)
            expression_parts.append(stem)
            previous_was_term = True

        corrected = []
        corrections = {}
        unknown = []
        for term in keywords:
            corrected_term = self.correct_term(term)
            if corrected_term is None:
                corrected.append(term)
                unknown.append(term)
            else:
                corrected.append(corrected_term)
                if corrected_term != term:
                    corrections[term] = corrected_term

        boolean_expression = " ".join(
            corrections.get(part, part) if part not in {"AND", "OR", "NOT"} else part
            for part in expression_parts
        )

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            keywords=keywords,
            corrected_keywords=corrected,
            corrections=corrections,
            unknown_terms=unknown,
            operators=operators,
            field_filter=field_filter,
            boolean_expression=boolean_expression or None,
            phrases=phrases,
        )

    def _extract_field_filter(self, normalized_query: str) -> str | None:
        if re.search(r"\b(titre|nom)\b", normalized_query):
            return "name"
        if re.search(r"\b(ingredient|ingredients|ingr)\b", normalized_query):
            return "ingredients"
        if re.search(r"\b(etape|etapes|preparation|instruction|instructions)\b", normalized_query):
            return "steps"
        if re.search(r"\b(personne|personnes|serving|servings)\b", normalized_query):
            return "metadata"
        return None

    def correct_term(
        self,
        term: str,
        *,
        prefix_min: int = 2,
        prefix_max: int = 5,
        max_distance: int = 2,
    ) -> str | None:
        """Correct a query term using TD4 prefix candidates and Levenshtein."""
        if NUMBER_RE.match(term) or term in self.lexicon:
            return term

        candidates: set[str] = set()
        for size in range(min(prefix_max, len(term)), prefix_min - 1, -1):
            prefix = term[:size]
            candidates.update(item for item in self.lexicon if item.startswith(prefix))
            if candidates:
                break

        # Robust fallback for prefix errors. TD4 asks to identify the weakness
        # of prefix-only correction; this keeps the engine usable for typos at
        # the beginning of the word.
        if not candidates:
            candidates = {
                item
                for item in self.lexicon
                if abs(len(item) - len(term)) <= max_distance
            }

        if not candidates:
            return None

        best = min(candidates, key=lambda item: (levenshtein(term, item), -self.collection_frequency(item), item))
        return best if levenshtein(term, best) <= max_distance else None

    def collection_frequency(self, term: str) -> int:
        return sum(self.inverted_indexes["all"].get(term, {}).values())

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: str = "ranked",
        field_name: str | None = None,
        explain: bool = False,
    ) -> tuple[list[SearchResult], QueryAnalysis]:
        analysis = self.analyze_query(query)
        selected_field = field_name or analysis.field_filter or "all"
        if selected_field not in FIELDS:
            raise ValueError(f"Unknown field: {selected_field}")

        if mode == "boolean":
            candidate_ids = self.boolean_search(analysis, selected_field)
        elif mode == "hybrid":
            candidate_ids = self.boolean_search(analysis, selected_field) or set()
        else:
            candidate_ids = None

        results = self.ranked_search(
            analysis.corrected_keywords,
            top_k=top_k,
            field_name=selected_field,
            candidate_ids=candidate_ids,
        )
        if mode == "boolean" and not analysis.corrected_keywords:
            results = [
                self._result_for_doc(doc_id, 0.0, [], analysis.corrected_keywords)
                for doc_id in sorted(candidate_ids or [])
            ][:top_k]

        if explain:
            self._print_analysis(analysis, selected_field, mode)
        return results, analysis

    def ranked_search(
        self,
        query_terms: list[str],
        *,
        top_k: int,
        field_name: str,
        candidate_ids: set[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[SearchResult]:
        if not query_terms:
            return []

        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, set[str]] = defaultdict(set)
        candidate_ids = set(candidate_ids) if candidate_ids is not None else None

        field_weights = {"name": 1.8, "ingredients": 1.25, "steps": 1.0, "metadata": 1.0, "all": 1.0}
        index = self.inverted_indexes[field_name]
        avg_len = max(self.avg_doc_len[field_name], 1e-9)

        for term in query_terms:
            postings = index.get(term, {})
            for doc_id, tf in postings.items():
                if candidate_ids is not None and doc_id not in candidate_ids:
                    continue
                doc_len = self.doc_lengths[field_name].get(doc_id, 0)
                denom = tf + k1 * (1 - b + b * doc_len / avg_len)
                scores[doc_id] += self.idf[field_name].get(term, 0.0) * (tf * (k1 + 1)) / denom
                matched[doc_id].add(term)

        results = []
        for doc_id, score in scores.items():
            doc = self.doc_by_id[doc_id]
            section = doc.metadata.get("section_type", "")
            coverage = len(matched[doc_id]) / max(len(set(query_terms)), 1)
            score *= 1.0 + coverage
            if field_name == "all":
                score *= field_weights.get(section, 1.0)
            else:
                score *= field_weights.get(field_name, 1.0)
            results.append(self._result_for_doc(doc_id, score, sorted(matched[doc_id]), query_terms))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def boolean_search(self, analysis: QueryAnalysis, field_name: str = "all") -> set[str]:
        if not analysis.boolean_expression:
            return set()
        parser = BooleanQueryParser(analysis.boolean_expression.split())
        tree = parser.parse()
        return self._eval_boolean_tree(tree, field_name)

    def _eval_boolean_tree(self, node, field_name: str) -> set[str]:
        kind = node[0]
        if kind == "TERM":
            return set(self.inverted_indexes[field_name].get(node[1], {}))
        if kind == "AND":
            return self._eval_boolean_tree(node[1], field_name) & self._eval_boolean_tree(node[2], field_name)
        if kind == "OR":
            return self._eval_boolean_tree(node[1], field_name) | self._eval_boolean_tree(node[2], field_name)
        if kind == "NOT":
            return self.all_doc_ids - self._eval_boolean_tree(node[1], field_name)
        raise ValueError(f"Unknown boolean node: {node}")

    def matching_chunks(self, terms: Iterable[str], *, field_name: str = "all") -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for term in terms:
            for norm in analyze_tokens(term):
                corrected = self.correct_term(norm) or norm
                out[corrected] = sorted(self.inverted_indexes[field_name].get(corrected, {}))
        return out

    def _result_for_doc(
        self,
        doc_id: str,
        score: float,
        matched_terms: list[str],
        query_terms: list[str],
    ) -> SearchResult:
        doc = self.doc_by_id[doc_id]
        return SearchResult(
            chunk_id=doc_id,
            score=score,
            matched_terms=matched_terms,
            text=doc.text,
            metadata=doc.metadata,
            snippet=make_snippet(doc.text, query_terms),
        )

    def _print_analysis(self, analysis: QueryAnalysis, field_name: str, mode: str) -> None:
        print("Query analysis")
        print(f"  mode: {mode}")
        print(f"  field: {field_name}")
        print(f"  keywords: {analysis.keywords}")
        print(f"  corrected: {analysis.corrected_keywords}")
        if analysis.corrections:
            print(f"  corrections: {analysis.corrections}")
        if analysis.unknown_terms:
            print(f"  unknown: {analysis.unknown_terms}")
        if analysis.boolean_expression:
            print(f"  boolean: {analysis.boolean_expression}")

    def to_json_dict(self) -> dict:
        return {
            "version": INDEX_VERSION,
            "documents": [asdict(doc) for doc in self.documents],
            "anti_dictionary": sorted(self.anti_dictionary),
            "lexicon": sorted(self.lexicon),
            "token_to_stem": self.token_to_stem,
            "doc_lengths": self.doc_lengths,
            "reverse_dictionary": self.reverse_dictionary,
            "inverted_indexes": {
                field_name: {term: dict(postings) for term, postings in index.items()}
                for field_name, index in self.inverted_indexes.items()
            },
            "idf": self.idf,
            "avg_doc_len": self.avg_doc_len,
            "tf": self.tf,
            "tfidf": self.tfidf,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "KeywordSearchEngine":
        if payload.get("version") != INDEX_VERSION:
            raise ValueError("Index version mismatch")
        engine = cls([ChunkDocument(**doc) for doc in payload["documents"]])
        engine.anti_dictionary = set(payload["anti_dictionary"])
        engine.lexicon = set(payload["lexicon"])
        engine.token_to_stem = payload["token_to_stem"]
        engine.doc_lengths = payload["doc_lengths"]
        engine.reverse_dictionary = payload["reverse_dictionary"]
        engine.inverted_indexes = {
            field_name: defaultdict(dict, index)
            for field_name, index in payload["inverted_indexes"].items()
        }
        engine.idf = payload["idf"]
        engine.avg_doc_len = payload["avg_doc_len"]
        engine.tf = payload["tf"]
        engine.tfidf = payload["tfidf"]
        return engine

    def save(self, path: Path = KEYWORD_INDEX_JSON) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = KEYWORD_INDEX_JSON) -> "KeywordSearchEngine":
        return cls.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


class KeywordSearchIndex(KeywordSearchEngine):
    """Backward-compatible name from the first prototype."""


class BooleanQueryParser:
    """Parser for TERM, AND, OR, NOT with simple precedence."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse(self):
        if not self.tokens:
            return ("TERM", "")
        return self._parse_or()

    def _parse_or(self):
        node = self._parse_and()
        while self._peek() == "OR":
            self._next()
            node = ("OR", node, self._parse_and())
        return node

    def _parse_and(self):
        node = self._parse_not()
        while self._peek() == "AND":
            self._next()
            node = ("AND", node, self._parse_not())
        return node

    def _parse_not(self):
        if self._peek() == "NOT":
            self._next()
            return ("NOT", self._parse_not())
        token = self._next()
        return ("TERM", token or "")

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str | None:
        token = self._peek()
        if token is not None:
            self.pos += 1
        return token


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, ch_left in enumerate(left, start=1):
        current = [i]
        for j, ch_right in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ch_left != ch_right),
                )
            )
        previous = current
    return previous[-1]


def make_snippet(text: str, query_terms: list[str], max_chars: int = 260) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    normalized = normalize_text(compact)
    positions = []
    for term in query_terms:
        pos = normalized.find(term)
        if pos >= 0:
            positions.append(pos)
    start = max(min(positions) - 70, 0) if positions else 0
    end = min(start + max_chars, len(compact))
    snippet = compact[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(compact):
        snippet += "..."
    return snippet


def build_index(
    chunks_path: Path = CHUNKS_JSONL,
    index_path: Path = KEYWORD_INDEX_JSON,
) -> KeywordSearchEngine:
    engine = KeywordSearchEngine(load_chunks(chunks_path))
    engine.save(index_path)
    return engine


def load_or_build_index(
    chunks_path: Path = CHUNKS_JSONL,
    index_path: Path = KEYWORD_INDEX_JSON,
    *,
    rebuild: bool = False,
) -> KeywordSearchEngine:
    if rebuild or not index_path.exists():
        return build_index(chunks_path, index_path)
    try:
        return KeywordSearchEngine.load(index_path)
    except ValueError:
        return build_index(chunks_path, index_path)


def evaluate(
    engine: KeywordSearchEngine,
    eval_path: Path,
    *,
    top_k_values: tuple[int, ...] = (1, 3, 5, 10),
    mode: str = "ranked",
) -> dict:
    """Evaluate local recall/precision from a JSONL ground-truth file.

    Each line can contain:

        {"query": "...", "expected_chunk_ids": ["..."]}
        {"query": "...", "expected_recipe": "Risotto aux ..."}
        {"query": "...", "expected_recipes": ["..."]}
    """
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = {
        f"recall@{k}": 0 for k in top_k_values
    } | {
        f"precision@{k}": 0.0 for k in top_k_values
    }
    timings = []
    details = []

    for row in rows:
        query = row["query"]
        start = time.perf_counter()
        results, _ = engine.search(query, top_k=max(top_k_values), mode=mode)
        timings.append(time.perf_counter() - start)

        expected_ids = set(row.get("expected_chunk_ids") or [])
        expected_recipes = set()
        if row.get("expected_recipe"):
            expected_recipes.add(row["expected_recipe"])
        expected_recipes.update(row.get("expected_recipes") or [])

        def is_relevant(result: SearchResult) -> bool:
            return result.chunk_id in expected_ids or result.metadata.get("recipe_name") in expected_recipes

        relevances = [is_relevant(result) for result in results]
        for k in top_k_values:
            top = relevances[:k]
            hit = any(top)
            metrics[f"recall@{k}"] += int(hit)
            metrics[f"precision@{k}"] += (sum(top) / k)
        details.append({"query": query, "hits": relevances, "top_ids": [r.chunk_id for r in results]})

    n = max(len(rows), 1)
    for k in top_k_values:
        metrics[f"recall@{k}"] /= n
        metrics[f"precision@{k}"] /= n

    return {
        "n_queries": len(rows),
        "metrics": metrics,
        "avg_time_ms": (sum(timings) / max(len(timings), 1)) * 1000,
        "details": details,
    }


def analyze_query(
    query: str,
    *,
    index_path: Path = KEYWORD_INDEX_JSON,
    chunks_path: Path = CHUNKS_JSONL,
    rebuild: bool = False,
) -> dict:
    """Analyze a natural-language query and return a serializable structure.

    This replaces the separate TD5 wrapper and is useful for debugging the
    keyword side of a future hybrid retriever.
    """
    engine = load_or_build_index(chunks_path, index_path, rebuild=rebuild)
    return asdict(engine.analyze_query(query))


def keyword_search(
    query: str,
    *,
    top_k: int = 5,
    mode: str = "ranked",
    field_name: str | None = None,
    index_path: Path = KEYWORD_INDEX_JSON,
    chunks_path: Path = CHUNKS_JSONL,
    rebuild: bool = False,
) -> dict:
    """Run keyword retrieval and return data ready to merge with embeddings.

    Output shape is intentionally close to what vector retrieval should return:
    each result carries a stable id, score, content, metadata, and matched
    keyword terms. Hybrid code can normalize scores and fuse this list with
    vector-search results.
    """
    engine = load_or_build_index(chunks_path, index_path, rebuild=rebuild)
    results, analysis = engine.search(
        query,
        top_k=top_k,
        mode=mode,
        field_name=field_name,
    )
    return {
        "query": query,
        "analysis": asdict(analysis),
        "results": [
            {
                "id": result.chunk_id,
                "score": result.score,
                "text": result.text,
                "metadata": result.metadata,
                "matched_terms": result.matched_terms,
                "snippet": result.snippet,
                "retriever": "keyword",
            }
            for result in results
        ],
    }


def _print_results(results: list[SearchResult]) -> None:
    for i, result in enumerate(results, start=1):
        name = result.metadata.get("recipe_name", "")
        section = result.metadata.get("section_type", "")
        terms = ", ".join(result.matched_terms)
        print(f"{i}. {name} [{section}] score={result.score:.3f} terms={terms}")
        print(f"   id: {result.chunk_id}")
        print(f"   source: {result.metadata.get('source_url', '')}")
        print(f"   {result.snippet}")


def _interactive(engine: KeywordSearchEngine, *, mode: str, top_k: int) -> None:
    print("Recipe keyword search. Empty query exits.")
    while True:
        query = input("> ").strip()
        if not query:
            break
        results, _ = engine.search(query, top_k=top_k, mode=mode, explain=True)
        _print_results(results)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="TD-style keyword search engine for recipe chunks.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_JSONL, help="Path to chunks.jsonl")
    parser.add_argument("--index", type=Path, default=KEYWORD_INDEX_JSON, help="Path to index JSON")
    parser.add_argument("--query", type=str, default="", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--mode", choices=("ranked", "boolean", "hybrid"), default="ranked")
    parser.add_argument("--field", choices=FIELDS, default=None, help="Restrict search to one field")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index even if it exists")
    parser.add_argument("--build-only", action="store_true", help="Build index and exit")
    parser.add_argument("--explain", action="store_true", help="Print query analysis")
    parser.add_argument("--interactive", action="store_true", help="Start terminal search UI")
    parser.add_argument("--eval", type=Path, default=None, help="Evaluate with a JSONL ground-truth file")
    args = parser.parse_args()

    engine = load_or_build_index(args.chunks, args.index, rebuild=args.rebuild)
    print(
        f"Index ready: {args.index} "
        f"({len(engine.documents)} chunks, {len(engine.inverted_indexes['all'])} terms, "
        f"{len(engine.anti_dictionary)} stop terms)"
    )

    if args.build_only:
        return
    if args.eval:
        print(json.dumps(evaluate(engine, args.eval, mode=args.mode), ensure_ascii=False, indent=2))
        return
    if args.interactive:
        _interactive(engine, mode=args.mode, top_k=args.top_k)
        return
    if args.query:
        results, _ = engine.search(
            args.query,
            top_k=args.top_k,
            mode=args.mode,
            field_name=args.field,
            explain=args.explain,
        )
        _print_results(results)
        return
    print('Pass --query, for example: --query "risotto champignons parmesan"')


if __name__ == "__main__":
    main()
