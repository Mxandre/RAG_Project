# LO17 / AI31 Recipe RAG 项目

[中文](README.md) | [Français](README.fr.md)

本项目实现了一个面向法语菜谱语料的 RAG（Retrieval Augmented Generation）系统。主题数据来自 [jemangefrancais.com](https://jemangefrancais.com)，系统支持关键词检索、向量检索、混合检索、可选 Gemini 生成、RAG 评估、幻觉控制与 Streamlit 演示界面。

项目对应 `Consignes pour le Projet.pdf` 中的要求：

- 选择一个有动机的主题，并使用法语文档数据。
- 实现基于 LangChain / Chroma / Gemini 的 RAG。
- 提供 RAG 评估和 hallucination 管理。
- 提供 Streamlit 应用。
- 提供可复现源码。

## 当前数据

仓库中已包含处理后的共享数据：

- `data/recipes.jsonl`：112 条菜谱记录。
- `data/chunks.jsonl`：220 个检索 chunk。
- `data/keyword_eval_queries.jsonl`：8 条评估查询。

每个 chunk 包含：

```python
{
    "id": "...",
    "text": "...",
    "metadata": {
        "recipe_name": "...",
        "source_url": "...",
        "section_type": "ingredients | steps",
        "servings": "..."
    }
}
```

## 项目结构

```text
RAG_Project/
├── data_process.py              # Chroma 向量库构建
├── p1_ingestion.py              # 数据抓取、解析、清洗、切块
├── p2_keyword_retrieval.py      # 关键词检索 / BM25 / 布尔检索 / 查询分析
├── p3_hybrid_retrieval.py       # 向量检索 + RRF 混合检索 + 检索评估
├── p4_rag_generate.py           # Gemini RAG 生成、意图识别、安全拦截
├── p5_rag_evaluate.py           # faithfulness、correctness、安全评估
├── streamlit_app.py             # Streamlit 聊天式应用
├── web_app.py                   # 标准库 HTTP 本地聊天界面
├── requirements.txt             # Python 依赖
└── data/
    ├── recipes.jsonl
    ├── chunks.jsonl
    └── keyword_eval_queries.jsonl
```

## 安装

推荐使用虚拟环境。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux / macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 运行 Streamlit 应用

Windows：

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

Linux / macOS：

```bash
.venv/bin/streamlit run streamlit_app.py
```

打开：

```text
http://127.0.0.1:8501
```

应用支持三种检索模式：

- `keyword`：BM25 / 关键词倒排索引检索。
- `vector`：`BAAI/bge-m3` embeddings + Chroma 向量检索。
- `hybrid`：关键词检索和向量检索通过 RRF 融合。

## 可选 Gemini 生成

如果只演示本地检索，可以不配置 API key。若要启用生成回答，需要配置：

```bash
GEMINI_API_KEY=your_key_here
```

或在本地 `.env` 文件中写入：

```text
GEMINI_API_KEY=your_key_here
```

生成模块位于 `p4_rag_generate.py`，并包含以下安全约束：

- 只允许基于检索上下文回答。
- 上下文不足时明确说明无法从可用来源回答。
- 对 prompt injection、prompt leaking、jailbreak 类请求进行拦截。
- 对检索文档中的恶意指令行进行 sanitization。

## 命令行使用

关键词检索：

```powershell
.\.venv\Scripts\python.exe p2_keyword_retrieval.py --query "risotto champignons parmesan" --top-k 5
```

混合检索：

```powershell
.\.venv\Scripts\python.exe p3_hybrid_retrieval.py --query "recette ratatouille" --mode hybrid --top-k 5
```

RAG 生成：

```powershell
.\.venv\Scripts\python.exe p4_rag_generate.py --query "Je voudrais faire une ratatouille." --retrieval-mode hybrid --top-k 5
```

标准库本地 Web 应用：

```powershell
.\.venv\Scripts\python.exe web_app.py --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

## 评估

关键词检索评估：

```powershell
.\.venv\Scripts\python.exe p2_keyword_retrieval.py --eval data\keyword_eval_queries.jsonl --mode ranked
```

三种检索模式对比：

```powershell
.\.venv\Scripts\python.exe p3_hybrid_retrieval.py --eval data\keyword_eval_queries.jsonl --compare
```

RAG faithfulness / correctness / security：

```powershell
.\.venv\Scripts\python.exe p5_rag_evaluate.py --mode hybrid --top-k 5
.\.venv\Scripts\python.exe p5_rag_evaluate.py --security-only
```

本地验证结果：

| 模式 | Recall@1 | Precision@1 | MRR |
|---|---:|---:|---:|
| keyword | 1.000 | 1.000 | 1.000 |
| vector | 0.875 | 0.875 | 0.938 |
| hybrid | 1.000 | 1.000 | 1.000 |

安全评估结果：

```text
score = 1.0
```

测试覆盖 prompt injection、prompt leaking、jailbreak 和 hostile context sanitization。

## 复现注意事项

- `chroma_db/` 和 `.hf_cache/` 如果不存在，需要先构建或下载模型。
- 当前向量模型为 `BAAI/bge-m3`。
- `p3_hybrid_retrieval.py` 中 embeddings 默认使用本地缓存加载；如果新机器没有缓存，需要先联网准备 HuggingFace 模型。
- Gemini 生成只在设置 `GEMINI_API_KEY` 后可用。

构建 Chroma 向量库：

```powershell
.\.venv\Scripts\python.exe p3_hybrid_retrieval.py --build-vectorstore --jsonl data\chunks.jsonl
```

## 项目要求对应情况

| 要求 | 对应实现 |
|---|---|
| 法语文档主题 | 法语菜谱数据，`data/recipes.jsonl` |
| RAG | `p2_keyword_retrieval.py`、`p3_hybrid_retrieval.py`、`p4_rag_generate.py` |
| Evaluation du RAG | `p5_rag_evaluate.py`、`data/keyword_eval_queries.jsonl` |
| Gestion des hallucinations | grounded prompt、faithfulness score、安全过滤 |
| Streamlit application | `streamlit_app.py` |
| Code source reproductible | `requirements.txt`、命令行脚本、预处理数据 |

## 最终交付提醒

PDF 还要求：

- 最多 8 页的报告，并写明每个人贡献比例。
- 5 分钟 presentation + 5 分钟 demo。
- 部署是 bonus，不是必需项。

这些材料不完全属于代码仓库本身，需要在提交课程作业时单独准备。

## License

See [LICENSE](LICENSE).
