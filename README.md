# RAG_LO17 — 法语菜谱 RAG 项目

[English](README.md) · [Français](README.fr.md)

基于法语菜谱语料库（[jemangefrancais.com](https://jemangefrancais.com)）的检索增强生成项目。本仓库目前仅包含 **P1 数据摄取阶段**：从源站抓取、解析、清洗、切块。下游通过稳定的 `run() -> list[Chunk]` 接口消费，对接混合检索（向量 + BM25）。

## 环境与命令

需要 Python 3.x。宿主 Python 受 PEP-668 外部托管限制，必须使用本地 venv 安装依赖。

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python p1_ingestion.py                 # 全量抓取所有分类页
.venv/bin/python p1_ingestion.py --max-pages 1   # 调试：仅抓首页
.venv/bin/python p1_ingestion.py --delay 0.4 --max-chars 800
```

参数：
- `--max-pages`：分页上限（调试用）
- `--delay`：请求间隔秒数（限速，礼貌爬取）
- `--max-chars`：步骤分块阈值

## 流水线结构

三阶段，全部产物落到 `data/`，可重入幂等：

```
crawl  → data/raw/<slug>.html         （收集 URL → 下载，已存在跳过）
parse  → data/recipes.jsonl           （parse_recipe 调度三种策略）
chunk  → data/chunks.jsonl + List[Chunk]
```

`run()` 编排全部三阶段。`data/raw/` 不入 git；两个 `.jsonl` 作为共享产物入仓。

## 解析策略

调度器 `parse_recipe()` 将 HTML 分类后路由到对应策略：

- **multi**：含 ≥2 个编号 `<h3>` 子菜谱头的页面（例如 *Les délices du Risotto* 含 3 个独立菜谱）。
  按 `<h3>` 边界切片，每个子菜谱独立解析，输出 `slug#N` 子记录。
- **standard**：单菜谱主流布局。文档顺序单遍扫，标记规则：
  - h2/h3/h4/strong/b 头 + 行首锚定 `<p>` 行用作段落标记
  - 短冒号子标题（如 `Le brownie :`）作为原料子组标签
  - 含 `recette`/`façon` 的短冒号行切换到步骤段（girolles 模式）
  - 步骤标记扩展到序数式 `1ère étape` / `2ème étape`（cèpe 模式）
  - 无显式头的 `<li>` 簇默认归原料；原料段中遇非 dash 散文 `<p>` 隐式开启步骤
- **skip**：缺少容器 / 无显式关键词标记 / 无步骤段恢复 → 落 `data/unparsed.md` 并附原因

步骤列表入库前 `" ".join` 融合为单段散文，提升向量召回（应对 entremet 等 30+ 短 `<li>` 步骤的页面）。

详细统计与解析覆盖见 [`docs/p1_parsing_report.md`](docs/p1_parsing_report.md)；未解析文件清单见 [`data/unparsed.md`](data/unparsed.md)。

## 数据契约

`Chunk` 是 P1 对下游的稳定接口：

```python
@dataclass
class Chunk:
    id: str          # f"{slug}::{section_type}"，多块步骤追加 ::{idx}
    text: str        # 检索正文：BM25 与向量同时使用
    metadata: dict   # {recipe_name, source_url, section_type, [servings, difficulty]}
```

字段级切分，非定长窗口：每条菜谱出一个 `ingredients` 块 + 一个 `steps` 块（超阈值按步骤边界再切，`chunk_index` 标识）。`section_type` 让下游可以分别加权（原料偏 BM25 关键词命中，步骤偏向量语义）。

`recipes.jsonl` 额外保留 `meta` 字典（`prep_time`/`cook_time`/`rest_time`/`difficulty`/`servings`），不进检索正文。

## 网页查询界面

项目提供了一个类似 ChatGPT 的本地网页入口，用于测试关键词检索、向量检索和混合检索。

首次运行先安装依赖：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Windows PowerShell 下也可以使用：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

一键启动网页：

```powershell
.\.venv\Scripts\python.exe web_app.py --host 127.0.0.1 --port 8000
```

然后打开：

```text
http://127.0.0.1:8000
```

网页默认不调用外部大模型，只运行本地检索：

- `Keyword`：BM25 / 关键词倒排索引检索
- `Vector`：`BAAI/bge-m3` embedding + Chroma 向量检索
- `Hybrid`：关键词结果和向量结果用 RRF 融合，推荐默认使用

如果勾选 `Generate answer`，才会调用 `.env` 中配置的 `GEMINI_API_KEY`。团队演示时可以不勾选该选项，只展示本地 hybrid retrieval，避免使用个人付费 API。

## 许可证

见 [LICENSE](LICENSE)。
