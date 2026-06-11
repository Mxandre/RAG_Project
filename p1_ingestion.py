"""P1 Ingestion — chargement / nettoyage / découpage RAG.

Pipeline : crawl (HTML) -> parse (name/ingredients/steps) -> chunk.
Interface publique : run() -> list[Chunk].
Corpus : https://jemangefrancais.com/blog/categorie/recettes.html.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("p1_ingestion")

BASE = "https://jemangefrancais.com"
CATEGORY_URL = f"{BASE}/blog/categorie/recettes.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAGbot/0.1)"}

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RECIPES_JSONL = DATA_DIR / "recipes.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
UNPARSED_MD = DATA_DIR / "unparsed.md"
PARSING_REPORT_MD = Path("docs") / "p1_parsing_report.md"

# Marqueurs de section. Mise en page hétérogène : h2/h3/strong/p/div.
_RE_INGRED = re.compile(r"ingr[ée]dient", re.I)
# Étapes : mots-clés + ordinaux "1ère étape" / "2ème étape" / "3e étape".
_RE_STEP_ORD = r"\d+\s*(?:ère|ere|ème|eme|e)\s+étape"
_RE_STEP = re.compile(rf"pr[ée]paration|pr[ée]parer|\bétapes?\b|\betapes?\b|montage|r[ée]alisation|instruction|{_RE_STEP_ORD}", re.I)
# Version ancrée (début de ligne) pour <p>/<li> longs : évite faux positifs en milieu de phrase.
_RE_ANCHOR = re.compile(rf"^\W*(?:ingr[ée]dient|pr[ée]paration|pr[ée]parer|étapes?|etapes?|montage|r[ée]alisation|instruction|{_RE_STEP_ORD})", re.I)
# Sous-titres numérotés multi-recettes, ex. "1. Risotto aux champignons sauvages".
_RE_SUBHEAD_NUM = re.compile(r"^\s*\d+\.\s+\w", re.I)
# Indice d'étapes implicites : titre court à deux points contenant "recette/façon".
_RE_STEP_HINT = re.compile(r"\brecette\b|\bfa[cç]on\b", re.I)
# Titres de fin/hors-sujet — ferment la section courante (évite conclusion/conseils en prose).
# Ancrage \b : évite "déconseillons"/"déconseillée" en plein texte d'étape.
_RE_END = re.compile(r"\b(?:conclusion|conseils?|astuces?|variantes?|le saviez|bon app|buon app|pour conclure|pourquoi)\b", re.I)
# Lignes metadata — extraites en champ propre, hors sections. Évite "Temps de préparation" en step.
_RE_PREP_TIME = re.compile(r"temps de pr[ée]paration\s*:?\s*(.+)", re.I)
_RE_COOK_TIME = re.compile(r"temps de cuisson\s*:?\s*(.+)", re.I)
_RE_REST_TIME = re.compile(r"temps de repos\s*:?\s*(.+)", re.I)
# Étiquettes nues : "Préparation : 15 minutes" / "Cuisson : 10 min" / "Marinade : 30 minutes".
# metadata seulement si la valeur après ":" est une durée pure (distingue du titre de section).
_DUR = r"\d+\s*(?:[àa-]\s*\d+\s*)?(?:min|minutes?|mn|h|heures?|jours?|secondes?)\b"
_RE_TIME_INLINE = re.compile(
    rf"^(?P<label>pr[ée]paration|cuisson|marinade|repos|r[ée]frig[ée]ration)\b[^:]*:\s*{_DUR}", re.I)
_RE_DIFFICULTY = re.compile(r"difficult[ée]\s*:?\s*(.+)", re.I)
_RE_SERVINGS = re.compile(r"pour\s+(\d[\d\s àa-]*?(?:personne|pot|part|convive)\w*)", re.I)
_RE_META_LINE = re.compile(r"temps de (pr[ée]paration|cuisson|repos)|^difficult", re.I)
# Lignes "durée pure" ("25 minutes" / "1 h 30") — valeur détachée d'une étiquette, bruit.
_RE_PURE_DUR = re.compile(r"^\d+\s*(?:[àa-]\s*\d+\s*)?(min|minutes?|mn|h|heures?|jours?|secondes?)\b\.?$", re.I)


@dataclass
class Chunk:
    id: str          # f"{slug}::{section_type}" (suffixe ::{idx} si steps multi-blocs)
    text: str        # corps indexé : BM25 + embedding vectoriel
    metadata: dict   # {recipe_name, source_url, section_type, [servings, difficulty]}


# --------------------------------------------------------------------------- #
# Étape 1 : crawl
# --------------------------------------------------------------------------- #
def _page_url(n: int) -> str:
    return CATEGORY_URL if n == 1 else f"{BASE}/blog/categorie/{n}/recettes.html"


def _slug_from_article_url(url: str) -> str:
    m = re.search(r"/blog/article/(.+?)\.html", url)
    return m.group(1) if m else re.sub(r"\W+", "-", url).strip("-")


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def collect_article_urls(max_pages: int | None = None, delay: float = 1.0) -> list[str]:
    """Parcourt toutes les pages de la catégorie, collecte les URLs d'articles (dédupliquées)."""
    seen: set[str] = set()
    ordered: list[str] = []
    n = 1
    while max_pages is None or n <= max_pages:
        url = _page_url(n)
        try:
            html = _get(url).text
        except requests.HTTPError as e:
            log.info("pagination arrêtée @ page %d (%s)", n, e)
            break
        soup = BeautifulSoup(html, "html.parser")
        page_links = [
            urljoin(BASE, a["href"])
            for a in soup.find_all("a", href=True)
            if "/blog/article/" in a["href"]
        ]
        new = [u for u in dict.fromkeys(page_links) if u not in seen]
        if not new:
            log.info("pagination arrêtée @ page %d (aucun nouveau lien)", n)
            break
        for u in new:
            seen.add(u)
            ordered.append(u)
        log.info("page %d : +%d articles (total %d)", n, len(new), len(ordered))
        n += 1
        time.sleep(delay)
    return ordered


def crawl(max_pages: int | None = None, raw_dir: Path = RAW_DIR, delay: float = 1.0) -> list[Path]:
    """Télécharge le HTML des articles dans raw_dir. Saute les fichiers existants (incrémental)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    urls = collect_article_urls(max_pages=max_pages, delay=delay)
    paths: list[Path] = []
    for url in urls:
        slug = _slug_from_article_url(url)
        path = raw_dir / f"{slug}.html"
        if path.exists():
            paths.append(path)
            continue
        try:
            html = _get(url).text
        except requests.HTTPError as e:
            log.warning("échec téléchargement %s (%s)", url, e)
            continue
        path.write_text(html, encoding="utf-8")
        paths.append(path)
        log.info("téléchargé %s", slug)
        time.sleep(delay)
    log.info("crawl terminé : %d HTML", len(paths))
    return paths


# --------------------------------------------------------------------------- #
# Étape 2 : parse
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _section_of(text: str) -> str | None:
    if _RE_INGRED.search(text):
        return "ingredients"
    if _RE_STEP.search(text):
        return "steps"
    return None


def _extract_name(soup: BeautifulSoup) -> str:
    """Nom de la recette : og:title prioritaire, fallback <h1> / <title> (suffixe site nettoyé)."""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return _clean(og["content"])
    h1 = soup.find("h1")
    if h1 and _clean(h1.get_text()):
        return _clean(h1.get_text())
    if soup.title and soup.title.string:
        return _clean(re.sub(r"\s*-\s*jemangefrancais\.com.*$", "", soup.title.string, flags=re.I))
    return ""


_TIME_LABEL_KEY = {"préparation": "prep_time", "preparation": "prep_time",
                   "cuisson": "cook_time", "marinade": "rest_time", "repos": "rest_time",
                   "réfrigération": "rest_time", "refrigeration": "rest_time"}


def _try_meta(text: str, meta: dict) -> bool:
    """Ligne metadata → extraite dans meta. Retourne True si capturée (l'appelant saute la ligne)."""
    for key, rx in (("prep_time", _RE_PREP_TIME), ("cook_time", _RE_COOK_TIME),
                    ("rest_time", _RE_REST_TIME), ("difficulty", _RE_DIFFICULTY)):
        m = rx.match(text)
        if m and re.search(r"\w", m.group(1)):  # valeur non vide (rejette "Temps de préparation :" seul)
            meta.setdefault(key, _clean(m.group(1)))
            return True
    m = _RE_TIME_INLINE.match(text)  # étiquette nue : "Préparation : 15 minutes"
    if m:
        key = _TIME_LABEL_KEY.get(m.group("label").lower())
        if key:
            meta.setdefault(key, _clean(re.sub(r"^[^:]*:\s*", "", text)))
        return True
    if _RE_META_LINE.search(text) or _RE_PURE_DUR.match(text):
        return True  # mot-clé metadata sans valeur ou durée pure détachée — sauté
    sv = _RE_SERVINGS.search(text)
    if sv and len(text) < 45 and not _RE_INGRED.search(text):
        # Ligne courte de portions ("Recette pour 4 à 6 personnes"). Exclut "Ingrédients pour N personnes :"
        # qui contient aussi "ingrédient", sinon vol du marqueur de section (bug girolles / riz-rouge).
        meta.setdefault("servings", _clean(sv.group(0)))
        return True
    return False


def _walk(descendants, *, default_ingredients: bool = True) -> tuple[list[str], list[str], dict, bool]:
    """Parcours en un passage : extrait ingredients/steps/meta. Bool final = marqueur explicite vu.

    Marqueur explicite = section_of (ingrédient/préparation/étape/...) ou _RE_STEP_HINT (recette/façon).
    Règles :
    - Titres/strong/b : mot-clé → change la section ; h2 sans mot-clé = fermeture dure.
    - Feuilles p/li/div : ancrage début de ligne ou titre à deux points → change la section.
    - Sous-titre court à deux points (`Le brownie :`), current=None → entre dans ingredients (sous-groupes entremet).
    - <li> ou <p> en prose sans section → fallback ingredients (cepe-bordeaux sans en-tête explicite).
    - current=ingredients + <p> long sans tiret → bascule vers steps (riz-rouge sans marqueur d'étapes).
    """
    ingredients: list[str] = []
    steps: list[str] = []
    meta: dict = {}
    current: str | None = None
    explicit_seen = False

    def add(txt: str) -> None:
        if txt and current:
            (ingredients if current == "ingredients" else steps).append(txt)

    for el in descendants:
        if not isinstance(el, Tag):
            continue
        tag = el.name

        if tag in ("h2", "h3", "h4", "strong", "b"):
            head = _clean(el.get_text())
            sec = _section_of(head)
            if sec:
                current = sec
                explicit_seen = True
            elif _RE_END.search(head):
                current = None
            elif tag == "h2":
                current = None
            continue

        if tag in ("p", "li", "div"):
            if el.find(["p", "li", "ul", "ol", "div", "table"]):
                continue
            txt = _clean(el.get_text(" "))
            if not txt:
                continue
            if _try_meta(txt, meta):
                continue

            sec = _section_of(txt)
            colon_short = txt.rstrip().endswith(":") and len(txt) < 80
            colon_with_kw = txt.rstrip().endswith(":") and sec is not None
            is_marker = bool(_RE_ANCHOR.match(txt)) or colon_short or colon_with_kw

            if sec and is_marker:
                current = sec
                explicit_seen = True
                rest = _clean(re.sub(r"^.*?:", "", txt, count=1)) if ":" in txt else ""
                if rest:
                    add(rest)
                continue

            # Sous-titre court à deux points sans mot-clé. Deux cas :
            # — contient "recette/façon" → début d'étapes (girolles "La recette de X :").
            # — sinon : étiquette de sous-groupe d'ingrédients (entremet "Le brownie :"/"Mousse chocolat :").
            if colon_short and sec is None and not _RE_END.search(txt):
                if _RE_STEP_HINT.search(txt):
                    current = "steps"
                    explicit_seen = True
                    continue
                if default_ingredients and current is None:
                    current = "ingredients"
                if current == "ingredients":
                    add(f"— {txt.rstrip(':').strip()}")
                    continue

            if _RE_END.search(txt):
                current = None
                continue

            # <p> prose en pleine section ingredients → bascule implicite vers steps (riz-rouge).
            if (tag == "p" and current == "ingredients"
                    and not txt.lstrip().startswith(("-", "–", "•", "—"))
                    and len(txt) > 50):
                current = "steps"

            # <li> sans section précédente → ingredients par défaut (cepe-bordeaux).
            if tag == "li" and current is None and default_ingredients:
                current = "ingredients"

            add(txt)

    return ingredients, steps, meta, explicit_seen


def _parse_standard(box: Tag, name: str, source_url: str) -> dict | None:
    ingredients, steps, meta, explicit_seen = _walk(box.descendants)

    if "servings" not in meta:
        sv = _RE_SERVINGS.search(box.get_text(" "))
        if sv:
            meta["servings"] = _clean(sv.group(0))

    if not name or not steps:
        # Section steps obligatoire — ingredients seuls (ex. crepes div-only) = mise en page pauvre, rejeté.
        return None
    if not explicit_seen:
        # Aucun marqueur explicite → tout vient des fallback li/p, typique des articles-liste (tapas/crepes).
        return None

    return {
        "slug": _slug_from_article_url(source_url),
        "name": name,
        "source_url": source_url,
        "meta": meta,
        "ingredients": ingredients,
        "steps": steps,
    }


def _multi_sub_headings(box: Tag) -> list[Tag]:
    """Liste des h3 numérotés sous-recettes (`1. Risotto aux ...`).

    Exclut les h3 de découpage d'étapes (`1. Préparation :` / `2. Cuisson :`) — ce sont des sections, pas des recettes.
    """
    out: list[Tag] = []
    for h in box.find_all("h3"):
        text = _clean(h.get_text())
        if not _RE_SUBHEAD_NUM.match(text):
            continue
        if _section_of(text) is not None:
            continue
        # Fin par ":" ("1. Préparation :", "2. Cuisson :") = sections d'étapes, pas sous-recettes.
        if text.rstrip().endswith(":"):
            continue
        out.append(h)
    return out


def _slice_after(start: Tag, stop: Tag | None):
    """Itère tous les descendants entre start et stop (parcours en profondeur)."""
    seen = False
    for el in start.parent.descendants if start.parent else []:
        if el is start:
            seen = True
            continue
        if not seen:
            continue
        if stop is not None and el is stop:
            break
        yield el


def _parse_multi(box: Tag, source_url: str) -> list[dict]:
    """Page multi-recettes — un sous-enregistrement par h3 numéroté, _walk indépendant pour chacun."""
    heads = _multi_sub_headings(box)
    base_slug = _slug_from_article_url(source_url)
    out: list[dict] = []
    for i, h in enumerate(heads, start=1):
        stop = heads[i] if i < len(heads) else None
        sub_name = _clean(re.sub(r"^\s*\d+\.\s*", "", h.get_text()))
        ingredients, steps, meta, _ = _walk(_slice_after(h, stop))
        if not ingredients and not steps:
            continue
        out.append({
            "slug": f"{base_slug}#{i}",
            "name": sub_name,
            "source_url": source_url,
            "meta": meta,
            "ingredients": ingredients,
            "steps": steps,
        })
    return out


def _classify(box: Tag) -> str:
    """multi = ≥ 2 h3 numérotés ; sinon standard (le parser juge ensuite s'il peut extraire)."""
    if len(_multi_sub_headings(box)) >= 2:
        return "multi"
    return "standard"


def parse_recipe(html: str, source_url: str) -> list[dict] | None:
    """Dispatcher : multi/standard/skip. None = saut ou échec."""
    soup = BeautifulSoup(html, "html.parser")
    name = _extract_name(soup)
    box = soup.select_one("div.blog_description") or soup.select_one("div.post-details")
    if box is None:
        log.warning("aucun conteneur de contenu : %s", source_url)
        return None

    kind = _classify(box)
    if kind == "multi":
        recs = _parse_multi(box, source_url)
        if recs:
            return recs
        log.warning("classé multi mais aucun sous-enregistrement, fallback standard : %s", source_url)

    rec = _parse_standard(box, name, source_url)
    if rec is None:
        log.warning("champs essentiels manquants, sauté : %s", source_url)
        return None
    return [rec]


def _url_from_raw_path(path: Path) -> str:
    return f"{BASE}/blog/article/{path.stem}.html"


def _skip_reason(html: str) -> str:
    """Diagnostic du rejet (pour le rapport)."""
    soup = BeautifulSoup(html, "html.parser")
    box = soup.select_one("div.blog_description") or soup.select_one("div.post-details")
    if box is None:
        return "no container (`div.blog_description` / `div.post-details`)"
    name = _extract_name(soup)
    _, _, _, explicit_seen = _walk(box.descendants)
    if not name:
        return "no recipe title"
    if not explicit_seen:
        return "no explicit ingrédient/préparation/étape marker (listicle or div-only layout)"
    return "no steps section recovered (ingredients-only / steps merged into prose)"


def build_recipes_jsonl(raw_dir: Path = RAW_DIR, out: Path = RECIPES_JSONL) -> Path:
    """Parse tous les HTML de raw_dir → recipes.jsonl. Écrit aussi unparsed.md + parsing_report.md."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_standard = 0
    n_multi_records = 0
    n_multi_pages = 0
    unparsed: list[tuple[str, str]] = []  # (slug, reason)

    with out.open("w", encoding="utf-8") as f:
        for path in sorted(raw_dir.glob("*.html")):
            n_total += 1
            html = path.read_text(encoding="utf-8")
            recs = parse_recipe(html, _url_from_raw_path(path))
            if not recs:
                unparsed.append((path.stem, _skip_reason(html)))
                continue
            for rec in recs:
                # Fusion des étapes : liste de <li> fragmentés → un seul paragraphe (meilleur rappel vectoriel).
                if rec.get("steps"):
                    rec["steps"] = [" ".join(rec["steps"])]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if len(recs) > 1:
                n_multi_pages += 1
                n_multi_records += len(recs)
            else:
                n_standard += 1
    log.info("parse terminé : %d standard + %d multi-records (depuis %d pages) ; %d sautés",
             n_standard, n_multi_records, n_multi_pages, len(unparsed))
    _emit_unparsed_md(unparsed)
    _emit_parsing_report(n_total, n_standard, n_multi_pages, n_multi_records, unparsed)
    return out


def _emit_unparsed_md(unparsed: list[tuple[str, str]]) -> None:
    UNPARSED_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Unparsed recipes\n",
             "Pages skipped by `parse_recipe` (heterogeneous layouts not yet supported).\n",
             "| File | Reason |", "|---|---|"]
    for slug, reason in sorted(unparsed):
        lines.append(f"| `{slug}.html` | {reason} |")
    UNPARSED_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_parsing_report(n_total: int, n_standard: int, n_multi_pages: int,
                         n_multi_records: int, unparsed: list[tuple[str, str]]) -> None:
    PARSING_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    n_parsed_pages = n_total - len(unparsed)
    lines = [
        "# P1 Parsing Report",
        "",
        "## Strategy",
        "",
        "Dispatcher in `parse_recipe()` classifies each HTML then routes to a strategy parser. A page is",
        "rejected (logged to `data/unparsed.md`) when it has no recipe container, no explicit",
        "ingrédient/préparation/étape marker (listicle / div-only layouts), or no steps section recovered.",
        "",
        "**multi** — page contains ≥2 numbered `<h3>` sub-recipe headings whose text is not colon-",
        "terminated and not a known step keyword (e.g. *Les délices du Risotto*: `1. Risotto aux",
        "champignons sauvages`). Each sub-recipe is sliced and parsed independently with `slug#N` ids.",
        "",
        "**standard** — single-recipe pages. Single document-order pass via `_walk()` collects",
        "ingredients/steps using section markers. Rules:",
        "",
        "- `_RE_INGRED` / `_RE_STEP` regex match on h2/h3/h4/strong/b headers + colon-anchored `<p>` lines.",
        "- Step markers extended to numbered ordinals: `1ère étape` / `2ème étape` / `3e étape`.",
        "- Servings regex no longer steals ingredient markers like `Ingrédients pour 4 personnes :`.",
        "- Short colon-terminated sub-titles (e.g. `Le brownie :`) act as ingredient sub-group labels;",
        "  ones containing `recette`/`façon` switch to the steps section instead (e.g. *girolles*",
        "  `La recette des girolles persillées :`).",
        "- `_RE_END` is word-boundary anchored so derivatives like `déconseillons` no longer close the",
        "  current section.",
        "- `<li>` with no preceding section marker defaults to `ingredients` (`cepe-bordeaux`).",
        "- Inside `ingredients`, a non-dash `<p>` longer than 50 chars implicitly opens steps",
        "  (`riz-rouge-de-camargue`).",
        "",
        "Step lists are fused into a single space-joined paragraph before persistence — better vector",
        "recall for fragmented procedural pages (e.g. *entremet* with 30+ tiny `<li>` steps).",
        "",
        "## Coverage",
        "",
        f"- Total HTML files: **{n_total}**",
        f"- Parsed pages: **{n_parsed_pages}** ({n_parsed_pages * 100 // n_total}%)",
        f"  - standard: {n_standard}",
        f"  - multi-recipe pages: {n_multi_pages} → {n_multi_records} sub-records",
        f"- Skipped: **{len(unparsed)}** (see `data/unparsed.md`)",
        "",
        "## Skipped files",
        "",
        "| File | Reason |",
        "|---|---|",
    ]
    for slug, reason in sorted(unparsed):
        lines.append(f"| `{slug}.html` | {reason} |")
    PARSING_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_recipes(path: Path = RECIPES_JSONL) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Étape 3 : chunk
# --------------------------------------------------------------------------- #
def _split_steps(steps: list[str], max_chars: int) -> list[str]:
    """Agrège par frontière d'étape ; ouvre un nouveau bloc si cumul > max_chars."""
    blocks: list[str] = []
    cur: list[str] = []
    size = 0
    for s in steps:
        if cur and size + len(s) > max_chars:
            blocks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(s)
        size += len(s) + 1
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def chunk_recipes(recipes: list[dict], max_chars: int = 800, out: Path = CHUNKS_JSONL) -> list[Chunk]:
    """Découpage par champ : chaque recette → ingredients chunk + steps chunk(s). Persiste et renvoie."""
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []

    for r in recipes:
        slug, name, url = r["slug"], r["name"], r["source_url"]
        rmeta = r.get("meta") or {}
        # Propagation restreinte aux champs utiles pour filtrage/citation, garde la metadata légère.
        extra = {k: rmeta[k] for k in ("servings", "difficulty") if rmeta.get(k)}

        def _meta(section_type: str, **kw) -> dict:
            return {"recipe_name": name, "source_url": url,
                    "section_type": section_type, **extra, **kw}

        if r.get("ingredients"):
            chunks.append(Chunk(
                id=f"{slug}::ingredients",
                text="\n".join(r["ingredients"]),
                metadata=_meta("ingredients"),
            ))

        steps = r.get("steps") or []
        if steps:
            blocks = _split_steps(steps, max_chars)
            multi = len(blocks) > 1
            for i, block in enumerate(blocks):
                cid = f"{slug}::steps::{i}" if multi else f"{slug}::steps"
                meta = _meta("steps", chunk_index=i) if multi else _meta("steps")
                chunks.append(Chunk(id=cid, text=block, metadata=meta))

    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    log.info("chunk terminé : %d chunks -> %s", len(chunks), out)
    return chunks


# --------------------------------------------------------------------------- #
# Orchestration — interface principale
# --------------------------------------------------------------------------- #
def run(max_pages: int | None = None, delay: float = 1.0, max_chars: int = 800) -> list[Chunk]:
    """Enchaîne crawl -> parse -> chunk. Retourne List[Chunk] (consommé par le retrieval hybride vector+BM25)."""
    crawl(max_pages=max_pages, delay=delay)
    build_recipes_jsonl()
    recipes = load_recipes()
    return chunk_recipes(recipes, max_chars=max_chars)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="P1 ingestion pipeline")
    ap.add_argument("--max-pages", type=int, default=None, help="limite le nombre de pages de catégorie (debug)")
    ap.add_argument("--delay", type=float, default=1.0, help="intervalle entre requêtes en secondes (politesse)")
    ap.add_argument("--max-chars", type=int, default=800, help="seuil de découpe des chunks steps")
    args = ap.parse_args()

    result = run(max_pages=args.max_pages, delay=args.delay, max_chars=args.max_chars)
    print(f"\nTotal {len(result)} chunks")
    for c in result[:3]:
        print(f"\n[{c.id}] {c.metadata['section_type']}")
        print(c.text[:200])
