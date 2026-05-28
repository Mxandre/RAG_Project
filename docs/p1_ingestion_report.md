# P1 Ingestion 报告 — 加载/清洗/分块

数据源: https://jemangefrancais.com/blog/categorie/recettes.html （法语菜谱博客）
模块: `p1_ingestion.py` | 主输出接口: `run() -> list[Chunk]`
存储方案: 混合检索 向量 + BM25

---

## 1. Pipeline 概览

三阶段，全部产物落盘，可增量重跑：

```
crawl   分类页分页遍历 → 收集详情页 URL → 下载 HTML   → data/raw/<slug>.html
parse   每 HTML → 抽 name/ingredients/steps/meta      → data/recipes.jsonl
chunk   每 recipe → 字段切分 (超长步骤再切)            → data/chunks.jsonl + 返回 List[Chunk]
```

- 分页规律: 第 1 页 `/blog/categorie/recettes.html`，第 N 页 `/blog/categorie/N/recettes.html`，共 10 页
- 详情页规律: `/blog/article/<slug>.html`
- 限速 `time.sleep(delay)`；已存在 HTML 跳过（增量、幂等）

---

## 2. 数据字段定义与原因

### recipe 记录 (`data/recipes.jsonl`，每行一菜谱)

| 字段 | 类型 | 说明 | 为何这样定义 |
|------|------|------|-------------|
| `slug` | str | URL 末段 | 稳定唯一键，做 chunk id 前缀，去重/增量靠它 |
| `name` | str | 菜名 | 取 **og:title**（最干净），回退 h1/title。页面首个 h1 常是 logo/导航，不可靠 |
| `source_url` | str | 详情页 URL | 检索结果引用溯源（citation） |
| `meta` | dict | prep_time/cook_time/rest_time/difficulty/servings | 抽成独立字段：① 清洁正文（否则 "Temps de cuisson : 25 min" 污染 steps）② 给下游可过滤维度（按份数/难度筛） |
| `ingredients` | list[str] | 原料行 | 原料适合 **BM25 关键词**匹配（"farine"、"œuf"） |
| `steps` | list[str] | 步骤行 | 步骤适合 **向量语义**匹配（"怎么做 X"） |

### Chunk 对象 (主输出接口 `List[Chunk]`，`data/chunks.jsonl`)

```python
@dataclass
class Chunk:
    id: str          # f"{slug}::{section_type}" (步骤多块追加 ::{idx})
    text: str        # 检索正文: BM25 索引 + 向量 embed
    metadata: dict   # {recipe_name, source_url, section_type, [servings, difficulty]}
```

| 字段 | 为何这样定义 |
|------|-------------|
| `id` | 确定性可复现，多块步骤带 index 仍唯一。重跑不变 → 下游索引可增量更新 |
| `text` | 单一检索正文，同时喂 BM25 与向量编码，避免两套数据漂移 |
| `metadata.section_type` | `ingredients`\|`steps`，下游可分别加权（原料偏 BM25，步骤偏向量） |
| `metadata.recipe_name/source_url` | 引用与展示 |
| `metadata.servings/difficulty` | 仅透传高过滤价值字段，保持 chunk 精简（不塞全部 meta） |

### 分块策略：**按字段切**（每菜谱 ≥2 chunk）

- 原料 → 1 chunk；步骤 → 1 chunk（超 `max_chars=800` 按步骤边界再切，带 `chunk_index`）
- 原因：原料与步骤检索语义不同，分开 chunk 让混合检索各取所长；按步骤边界切而非定长滑窗，避免切断单步语义

---

## 3. 遇到的困难与解决方法

站点**无统一模板**，每页 HTML 结构不同。核心困难即异构。

### 困难 1 — 段标记标签不固定
原料/步骤标题出现在 `h2`/`h3`/`strong`/`b`/`p`/`div` 任意标签（普查: ing 标记 p=47 h2=45 h3=8 strong=4；step 标记 p=74 h2=27 ...）。
**解决**: 不靠固定 CSS 选择器。文档顺序单遍遍历，任意标签文本按正则判段落，状态机 `current` 跟踪当前段。

### 困难 2 — 内容载体 li 与 p 混用
原料: li 820 + p 401；步骤: p 937 + li 485。只收 li 会漏一半。
**解决**: li 与 p 同收，归入 `current` 段。

### 困难 3 — h2 大边界 vs h3 子标题
salade 用 `<h3>Pour la salade :` / `<h3>Pour la sauce :` 子标题（应保持段落）；而 `<h2>Conclusion` 应关闭段落。早期一刀切重置导致 salade 整页丢失。
**解决**: h2 = 硬边界（非 ingred/step 标题即关段）；h3/h4/strong = 软标记（仅命中正则才切换）。

### 困难 4 — 长正文中部含关键词被误判标记
`<p>La farine T65 ... pour la préparation du pain ...</p>` 含 "préparation"，被 `search` 误判为步骤标记 → 正文泄漏。
**解决**: 内容元素标记判定收紧 = 行首锚定（`^\W*(ingrédient|préparation|étape|...)`）**或** 冒号结尾短标题（`< 80` 字符）。标题（h2/h3/strong）仍用宽松 `search`（短、可信）。

### 困难 5 — 时间元数据污染 steps
两种形态：① "Temps de cuisson : 25 minutes" ② 裸标签 "Préparation : 15 minutes" ③ 标签值分离（"Temps de cuisson :" 一元素，"25 minutes" 另一元素）④ 带括号 "Préparation (la veille) : 25 minutes"。被当步骤 → "15 minutes" 泄漏。
**解决**: `_try_meta()` 优先识别：
- `Temps de (préparation|cuisson|repos) : <值>` → 对应 meta，要求值含实字符（防 "Temps de préparation :" 空值把 ":" 当值）
- 裸标签 `(préparation|cuisson|marinade|repos)[^:]*: <时长>` → 仅当冒号后是纯时长才算 meta（区别于 "Préparation :" 步骤段标题），`[^:]*` 容忍括号附加词
- 纯时长行 "25 minutes" → 直接跳过（分离值）

### 困难 6 — 结尾/旁支段落泄漏
"Conclusion" / "Difficultés" / "Pourquoi utiliser..." 等 h2 后正文混入 steps。
**解决**: `_RE_END`（conclusion/conseil/astuce/pourquoi/...）显式关段；h2 非 ingred/step 也兜底关段。

### 困难 7 — 菜名取错
`soup.find("h1")` 抓到页面首个无关 h1（曾输出 name="p"，且循环变量 `name=el.name` 覆盖 bug）。
**解决**: 改用 og:title；循环变量改名 `tag`。

---

## 4. 解析覆盖与未解析清单

| 指标 | 数值 |
|------|------|
| 下载 HTML | 113 |
| 成功解析菜谱 | 107 (95%) |
| 双段齐全 (原料+步骤) | 89 |
| 仅原料无步骤 | 5 |
| 仅步骤无原料 | 13 |
| 输出 chunks | 386 (原料 94 + 步骤 292) |
| 带 meta 菜谱 | 83 |

### 完全未解析 (6)

**无标准容器 (1)** — 页面非 `div.blog_description` 结构：
- `couscous-histoire-et-recette-on-vous-dit-tout-`

**关键字段缺失 (5)** — 整篇 prose、无原料/步骤结构，或非单菜谱：
- `tapas-origine-et-15-suggestions-de-recette` （"15 条建议"清单，非单菜谱）
- `recette-crepes-faciles-sans-repos` （整段散文，无列表/标题结构）
- `recette-de-girolles-poelee-de-girolles-persillees`
- `recette-du-cepe-de-bordeaux-a-la-poele-cepe-cuit-en-lamelles`
- `riz-rouge-de-camargue-aux-epinards-frais`

### 单段菜谱 (18) — 部分内容漏抓

**仅步骤无原料 (13)**: 原料无 "Ingrédients" 标题（如 mojito 纯 li 列原料），或原料引语过长非行首/无冒号（如 lasagnes "Voici donc la liste des ingrédients pour..."）。

**仅原料无步骤 (5)**: 步骤段无 "Préparation/Étape" 标题，步骤散文累积进 ingredients（如 cardon-lyonnais）。表现为 ingredients 含 >120 字符的句子型长行。

### 为何不继续修
剩余畸形页需逐页定制规则（无标题、纯散文、清单页）。放宽通用规则会反向引入误判（实测放宽原料引语 → +1 命中但 +19 prose 误入原料）。当前规则在覆盖率与精度间已达较优平衡，故止于此。

---

## 5. 后续可选改进

- 步骤无标题页: 启发式「ingredients 段出现句子型长行(>120c, 含动词) → 重分类为 steps」（有误判风险，需评估）
- 无原料标题页: 「步骤标题前的纯 li 块 → 推定为原料」（依赖列表顺序，脆弱）
- couscous 类: 增加备用容器选择器
- 原料行结构化: 量/单位/食材名拆分（利于精确过滤）

## 6. 运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python p1_ingestion.py                  # 全量
.venv/bin/python p1_ingestion.py --max-pages 1    # 调试 (前 1 分页)
```
参数: `--max-pages` 限分页数 | `--delay` 请求间隔秒 | `--max-chars` 步骤切分阈值
