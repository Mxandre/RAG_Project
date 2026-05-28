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

# 段落标记 (法语). 站点结构异构: 标记现于 h2/h3/strong/p/div.
_RE_INGRED = re.compile(r"ingr[ée]dient", re.I)
_RE_STEP = re.compile(r"pr[ée]paration|pr[ée]parer|\bétapes?\b|\betapes?\b|montage|r[ée]alisation|instruction", re.I)
# 锚定版 (行首): 用于内容元素长段落, 仅行首关键词才算标记, 防中部含词误判.
_RE_ANCHOR = re.compile(r"^\W*(ingr[ée]dient|pr[ée]paration|pr[ée]parer|étapes?|etapes?|montage|r[ée]alisation|instruction)", re.I)
# 结尾/旁支标题 — 关闭当前段落, 防 conclusion/conseils 等正文泄漏入 sections.
_RE_END = re.compile(r"conclusion|conseil|astuce|variante|le saviez|bon app|buon app|pour conclure|pourquoi", re.I)
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
    if sv and len(text) < 45:  # 短 servings 行 (如 "Recette pour 4 à 6 personnes")
        meta.setdefault("servings", _clean(sv.group(0)))
        return True
    return False


def parse_recipe(html: str, source_url: str) -> dict | None:
    """抽 name / ingredients[] / steps[] / meta. 关键字段缺失返回 None.

    站点结构异构 (见结构普查): 段标记现于 h2/h3/h4/strong/b/p/div, 内容载体 li 与 p 混用.
    策略: 文档顺序单遍; current 跟踪当前段落; h2=硬边界(非 ingred/step 即关段),
    h3/h4/strong/p=软标记(仅命中正则才切换); metadata 行抽独立字段不进 sections.
    """
    soup = BeautifulSoup(html, "html.parser")
    name = _extract_name(soup)

    box = soup.select_one("div.blog_description") or soup.select_one("div.post-details")
    if box is None:
        log.warning("无正文容器: %s", source_url)
        return None

    ingredients: list[str] = []
    steps: list[str] = []
    meta: dict = {}
    current: str | None = None

    def add(txt: str) -> None:
        if txt and current:
            (ingredients if current == "ingredients" else steps).append(txt)

    for el in box.descendants:
        if not isinstance(el, Tag):
            continue
        tag = el.name

        # 标记元素: 标题 h2/h3/h4 + 内联粗体 strong/b (本站常当段落标记).
        if tag in ("h2", "h3", "h4", "strong", "b"):
            head = _clean(el.get_text())
            sec = _section_of(head)
            if sec:
                current = sec
            elif _RE_END.search(head):
                current = None
            elif tag == "h2":
                current = None  # h2 硬边界: 非 ingred/step 标题 → 关闭段落
            continue

        # 内容元素 (叶子 p/li/div). 跳过含块级子元素者, 防父子文本重复收集.
        if tag in ("p", "li", "div"):
            if el.find(["p", "li", "ul", "ol", "div", "table"]):
                continue
            txt = _clean(el.get_text(" "))
            if not txt:
                continue
            if _try_meta(txt, meta):
                continue
            # 内联标记: 行首锚定 ("Étape 1 :"/"Ingrédients...") 或 冒号结尾短标题
            # ("Voici les ingrédients :"). 防长正文/含词原料行误判为段落标记.
            sec = _section_of(txt)
            is_marker = bool(_RE_ANCHOR.match(txt)) or (txt.rstrip().endswith(":") and len(txt) < 80)
            if sec and is_marker:
                current = sec
                rest = _clean(re.sub(r"^.*?:", "", txt, count=1)) if ":" in txt else ""
                if rest:
                    add(rest)
            elif _RE_END.search(txt):
                current = None
            else:
                add(txt)

    # servings 兜底: 全文扫一次 (常在标题/intro), 不覆盖 loop 内已抽值.
    if "servings" not in meta:
        sv = _RE_SERVINGS.search(box.get_text(" "))
        if sv:
            meta["servings"] = _clean(sv.group(0))

    if not name or (not ingredients and not steps):
        log.warning("关键字段缺失, 跳过: %s", source_url)
        return None

    return {
        "slug": _slug_from_article_url(source_url),
        "name": name,
        "source_url": source_url,
        "meta": meta,
        "ingredients": ingredients,
        "steps": steps,
    }


def _url_from_raw_path(path: Path) -> str:
    return f"{BASE}/blog/article/{path.stem}.html"


def build_recipes_jsonl(raw_dir: Path = RAW_DIR, out: Path = RECIPES_JSONL) -> Path:
    """解析 raw_dir 全部 HTML -> recipes.jsonl."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with out.open("w", encoding="utf-8") as f:
        for path in sorted(raw_dir.glob("*.html")):
            rec = parse_recipe(path.read_text(encoding="utf-8"), _url_from_raw_path(path))
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    log.info("parse 完成: %d recipes -> %s", n_ok, out)
    return out


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
