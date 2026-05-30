"""P1 Ingestion — RAG 数据加载/清洗/分块.

Pipeline: crawl (下载 HTML) -> parse (抽 name/ingredients/steps) -> chunk.
对外主接口: run() -> list[Chunk].
语料: https://jemangefrancais.com/blog/categorie/recettes.html (法语菜谱).
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

# 段落标记 (法语). 站点结构异构: 标记现于 h2/h3/strong/p/div.
_RE_INGRED = re.compile(r"ingr[ée]dient", re.I)
# 步骤标记: 关键词 + 序数式 "1ère étape" / "2ème étape" / "3e étape".
_RE_STEP_ORD = r"\d+\s*(?:ère|ere|ème|eme|e)\s+étape"
_RE_STEP = re.compile(rf"pr[ée]paration|pr[ée]parer|\bétapes?\b|\betapes?\b|montage|r[ée]alisation|instruction|{_RE_STEP_ORD}", re.I)
# 锚定版 (行首): 用于内容元素长段落, 仅行首关键词才算标记, 防中部含词误判.
_RE_ANCHOR = re.compile(rf"^\W*(?:ingr[ée]dient|pr[ée]paration|pr[ée]parer|étapes?|etapes?|montage|r[ée]alisation|instruction|{_RE_STEP_ORD})", re.I)
# 多菜谱 h3 编号头, e.g. "1. Risotto aux champignons sauvages".
_RE_SUBHEAD_NUM = re.compile(r"^\s*\d+\.\s+\w", re.I)
# 隐式步骤段提示: 短冒号标题含 "recette/façon" 多为 "La recette de X :" 步骤起点.
_RE_STEP_HINT = re.compile(r"\brecette\b|\bfa[cç]on\b", re.I)
# 结尾/旁支标题 — 关闭当前段落, 防 conclusion/conseils 等正文泄漏入 sections.
# 边界锚定: 防 "déconseillons"/"déconseillée" 等步骤正文里的派生词触发关段.
_RE_END = re.compile(r"\b(?:conclusion|conseils?|astuces?|variantes?|le saviez|bon app|buon app|pour conclure|pourquoi)\b", re.I)
# metadata 行 (非原料/步骤) — 抽成独立字段, 不进 sections. 也防 "Temps de préparation" 误判 step.
_RE_PREP_TIME = re.compile(r"temps de pr[ée]paration\s*:?\s*(.+)", re.I)
_RE_COOK_TIME = re.compile(r"temps de cuisson\s*:?\s*(.+)", re.I)
_RE_REST_TIME = re.compile(r"temps de repos\s*:?\s*(.+)", re.I)
# 裸标签时长行: "Préparation : 15 minutes" / "Cuisson : 10 min" / "Marinade : 30 minutes".
# 仅当冒号后是纯时长才算 metadata (区别于 "Préparation :" 步骤段标题).
_DUR = r"\d+\s*(?:[àa-]\s*\d+\s*)?(?:min|minutes?|mn|h|heures?|jours?|secondes?)\b"
_RE_TIME_INLINE = re.compile(
    rf"^(?P<label>pr[ée]paration|cuisson|marinade|repos|r[ée]frig[ée]ration)\b[^:]*:\s*{_DUR}", re.I)
_RE_DIFFICULTY = re.compile(r"difficult[ée]\s*:?\s*(.+)", re.I)
_RE_SERVINGS = re.compile(r"pour\s+(\d[\d\s àa-]*?(?:personne|pot|part|convive)\w*)", re.I)
_RE_META_LINE = re.compile(r"temps de (pr[ée]paration|cuisson|repos)|^difficult", re.I)
# 纯时长行 (如 "25 minutes" / "1 h 30") — 多为 "Temps de cuisson :" 标签的分离值, 噪声.
_RE_PURE_DUR = re.compile(r"^\d+\s*(?:[àa-]\s*\d+\s*)?(min|minutes?|mn|h|heures?|jours?|secondes?)\b\.?$", re.I)


@dataclass
class Chunk:
    id: str          # f"{slug}::{section_type}" (steps 多块时追加 ::{idx})
    text: str        # 检索正文: BM25 索引 + 向量 embed
    metadata: dict   # {recipe_name, source_url, section_type, [servings, difficulty]}


# --------------------------------------------------------------------------- #
# Stage 1: crawl
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
    """遍历分类页所有分页, 收集去重的详情页 URL."""
    seen: set[str] = set()
    ordered: list[str] = []
    n = 1
    while max_pages is None or n <= max_pages:
        url = _page_url(n)
        try:
            html = _get(url).text
        except requests.HTTPError as e:
            log.info("分页停止 @ page %d (%s)", n, e)
            break
        soup = BeautifulSoup(html, "html.parser")
        page_links = [
            urljoin(BASE, a["href"])
            for a in soup.find_all("a", href=True)
            if "/blog/article/" in a["href"]
        ]
        new = [u for u in dict.fromkeys(page_links) if u not in seen]
        if not new:
            log.info("分页停止 @ page %d (无新链接)", n)
            break
        for u in new:
            seen.add(u)
            ordered.append(u)
        log.info("page %d: +%d 详情页 (累计 %d)", n, len(new), len(ordered))
        n += 1
        time.sleep(delay)
    return ordered


def crawl(max_pages: int | None = None, raw_dir: Path = RAW_DIR, delay: float = 1.0) -> list[Path]:
    """下载全部详情页 HTML 到 raw_dir. 已存在则跳过 (增量). 返回本地路径列表."""
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
            log.warning("下载失败 %s (%s)", url, e)
            continue
        path.write_text(html, encoding="utf-8")
        paths.append(path)
        log.info("下载 %s", slug)
        time.sleep(delay)
    log.info("crawl 完成: %d HTML", len(paths))
    return paths


# --------------------------------------------------------------------------- #
# Stage 2: parse
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
    """菜名: og:title 最干净, 回退 <h1> / <title> (去站点后缀)."""
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
    """metadata 行 → 抽进 meta dict. 命中返回 True (调用方据此跳过, 不进 sections)."""
    for key, rx in (("prep_time", _RE_PREP_TIME), ("cook_time", _RE_COOK_TIME),
                    ("rest_time", _RE_REST_TIME), ("difficulty", _RE_DIFFICULTY)):
        m = rx.match(text)
        if m and re.search(r"\w", m.group(1)):  # 值含实字符 (排除 "Temps de préparation :" 无值)
            meta.setdefault(key, _clean(m.group(1)))
            return True
    m = _RE_TIME_INLINE.match(text)  # 裸标签时长 "Préparation : 15 minutes"
    if m:
        key = _TIME_LABEL_KEY.get(m.group("label").lower())
        if key:
            meta.setdefault(key, _clean(re.sub(r"^[^:]*:\s*", "", text)))
        return True
    if _RE_META_LINE.search(text) or _RE_PURE_DUR.match(text):
        return True  # metadata 关键词无值 / 分离的纯时长值 — 跳过, 防进 section
    sv = _RE_SERVINGS.search(text)
    if sv and len(text) < 45 and not _RE_INGRED.search(text):
        # 短 servings 行 (如 "Recette pour 4 à 6 personnes"). 排除 "Ingrédients pour N personnes :"
        # 这种同时含 ingrédient 关键词的标记, 否则会吞掉原料段标记 (girolles / riz-rouge bug).
        meta.setdefault("servings", _clean(sv.group(0)))
        return True
    return False


def _walk(descendants, *, default_ingredients: bool = True) -> tuple[list[str], list[str], dict, bool]:
    """单遍扫给定 descendants 序列, 抽 ingredients/steps/meta. 末位 bool=是否命中显式关键词标记.

    显式标记 = section_of 命中 (ingrédient/préparation/étape/...) 或 _RE_STEP_HINT (recette/façon).
    标记规则: 标题/strong/b 元素 — 命中关键词切段, h2 非命中=硬关段.
    内容元素 (p/li/div 叶子) — 锚定行首关键词 或 冒号结尾标题 → 切段.
    短冒号子标题 (`Le brownie :`) — current=None 时假设进入 ingredients (entremet 子组).
    `<li>` / 散文 `<p>` 出现而 current=None → fallback 入 ingredients (cepe-bordeaux 无显式 header).
    `current=ingredients` 下遇散文 `<p>` 无 dash 前缀 → 切到 steps (riz-rouge 无显式步骤标记).
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

            # 短冒号子标题无关键词. 区分两路:
            # — 含 "recette/façon" → 步骤段起点 (girolles "La recette de X :").
            # — 否则: 原料子组标签 (entremet "Le brownie :"/"Mousse chocolat :").
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

            # `<p>` 散文转步骤: 原料段中遇到非 dash 长散文 → 隐式步骤起点 (riz-rouge).
            if (tag == "p" and current == "ingredients"
                    and not txt.lstrip().startswith(("-", "–", "•", "—"))
                    and len(txt) > 50):
                current = "steps"

            # `<li>` fallback: 未声明段落 → 默认原料 (cepe-bordeaux).
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
        # 要求步骤段必须解析到 — 仅原料 (crepes div-only) 多为低质布局, 弃.
        return None
    if not explicit_seen:
        # 无任何显式关键词标记 → 全靠 fallback 抓的 li/p, 多为列表式文章 (tapas/crepes).
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
    """编号 h3 子菜谱头列表 (`1. Risotto aux ...`).

    排除步骤分节型 h3 (`1. Préparation :` / `2. Cuisson :`) — 它们是步骤章节而非独立菜谱.
    """
    out: list[Tag] = []
    for h in box.find_all("h3"):
        text = _clean(h.get_text())
        if not _RE_SUBHEAD_NUM.match(text):
            continue
        if _section_of(text) is not None:
            continue
        # 冒号终结 ("1. Préparation :", "2. Cuisson :") 是步骤章节, 非子菜谱.
        if text.rstrip().endswith(":"):
            continue
        out.append(h)
    return out


def _slice_after(start: Tag, stop: Tag | None):
    """yield start 之后到 stop 之前的所有 descendants (深度优先)."""
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
    """多菜谱页 — 每个编号 h3 切一个子记录, 各自运行 _walk."""
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
    """multi=≥2 编号 h3; 否则 standard (parser 自判能否抽出, None 时上层归 skip)."""
    if len(_multi_sub_headings(box)) >= 2:
        return "multi"
    return "standard"


def parse_recipe(html: str, source_url: str) -> list[dict] | None:
    """分类后路由: multi/standard/skip. None=skip 或解析失败."""
    soup = BeautifulSoup(html, "html.parser")
    name = _extract_name(soup)
    box = soup.select_one("div.blog_description") or soup.select_one("div.post-details")
    if box is None:
        log.warning("无正文容器: %s", source_url)
        return None

    kind = _classify(box)
    if kind == "multi":
        recs = _parse_multi(box, source_url)
        if recs:
            return recs
        log.warning("multi 分类但无子记录, 回退 standard: %s", source_url)

    rec = _parse_standard(box, name, source_url)
    if rec is None:
        log.warning("关键字段缺失, 跳过: %s", source_url)
        return None
    return [rec]


def _url_from_raw_path(path: Path) -> str:
    return f"{BASE}/blog/article/{path.stem}.html"


def _skip_reason(html: str) -> str:
    """诊断 None 返回原因 (报告用)."""
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
    """解析 raw_dir 全部 HTML → recipes.jsonl. 同时落 unparsed.md + parsing_report.md."""
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
                # 步骤融合: 碎片化 li 列表 → 单段散文, 利于向量召回.
                if rec.get("steps"):
                    rec["steps"] = [" ".join(rec["steps"])]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if len(recs) > 1:
                n_multi_pages += 1
                n_multi_records += len(recs)
            else:
                n_standard += 1
    log.info("parse 完成: %d standard + %d multi-records (from %d pages); %d skipped",
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
# Stage 3: chunk
# --------------------------------------------------------------------------- #
def _split_steps(steps: list[str], max_chars: int) -> list[str]:
    """按步骤边界聚合; 累计超 max_chars 则切新块."""
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
    """字段切: 每 recipe 出 ingredients chunk + steps chunk(超阈值再切). 落盘并返回."""
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []

    for r in recipes:
        slug, name, url = r["slug"], r["name"], r["source_url"]
        rmeta = r.get("meta") or {}
        # 仅透传可过滤/引用价值高的字段进 chunk metadata, 保持精简.
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
    log.info("chunk 完成: %d chunks -> %s", len(chunks), out)
    return chunks


# --------------------------------------------------------------------------- #
# Orchestration — 主输出接口
# --------------------------------------------------------------------------- #
def run(max_pages: int | None = None, delay: float = 1.0, max_chars: int = 800) -> list[Chunk]:
    """编排 crawl -> parse -> chunk. 返回 List[Chunk] (下游混合检索 vector+BM25 消费)."""
    crawl(max_pages=max_pages, delay=delay)
    build_recipes_jsonl()
    recipes = load_recipes()
    return chunk_recipes(recipes, max_chars=max_chars)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="P1 ingestion pipeline")
    ap.add_argument("--max-pages", type=int, default=None, help="限制分类页分页数 (调试)")
    ap.add_argument("--delay", type=float, default=1.0, help="请求间隔秒 (限速)")
    ap.add_argument("--max-chars", type=int, default=800, help="steps chunk 切分阈值")
    args = ap.parse_args()

    result = run(max_pages=args.max_pages, delay=args.delay, max_chars=args.max_chars)
    print(f"\n总计 {len(result)} chunks")
    for c in result[:3]:
        print(f"\n[{c.id}] {c.metadata['section_type']}")
        print(c.text[:200])
