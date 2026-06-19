# Projet RAG LO17 / AI31 sur des recettes françaises

Ce dépôt contient un système RAG (Retrieval Augmented Generation) appliqué à un corpus de recettes en français. Les données proviennent de [jemangefrancais.com](https://jemangefrancais.com). Le projet couvre la collecte des données, l'indexation, la recherche par mots-clés, la recherche vectorielle, la fusion hybride, la génération optionnelle avec Gemini, l'évaluation du RAG, la gestion des hallucinations et une application Streamlit.

Le projet répond aux consignes du fichier `Consignes pour le Projet.pdf` :

- choix motivé d'un sujet et de données en français ;
- code basé sur l'écosystème LangChain / Chroma / Gemini ;
- évaluation du RAG et gestion des hallucinations ;
- application avec Streamlit ;
- code source reproductible.

## Données disponibles

Les artefacts partagés sont inclus dans `data/` :

- `data/recipes.jsonl` : 112 recettes.
- `data/chunks.jsonl` : 220 extraits indexables.
- `data/keyword_eval_queries.jsonl` : 8 requêtes d'évaluation.

Chaque extrait suit la forme suivante :

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

## Structure du dépôt

```text
RAG_Project/
├── data_process.py              # Construction de la base vectorielle Chroma
├── p1_ingestion.py              # Crawl, parsing, nettoyage, chunking
├── p2_keyword_retrieval.py      # Recherche mots-clés / BM25 / booléen / analyse de requête
├── p3_hybrid_retrieval.py       # Recherche vectorielle + fusion RRF + évaluation retrieval
├── p4_rag_generate.py           # Génération Gemini, intention, sécurité
├── p5_rag_evaluate.py           # Faithfulness, correctness, sécurité
├── streamlit_app.py             # Application Streamlit en mode chat
├── web_app.py                   # Interface web locale avec la bibliothèque standard
├── requirements.txt             # Dépendances Python
└── data/
    ├── recipes.jsonl
    ├── chunks.jsonl
    └── keyword_eval_queries.jsonl
```

## Installation

Il est recommandé d'utiliser un environnement virtuel.

Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux / macOS :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Lancer l'application Streamlit

Windows :

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

Linux / macOS :

```bash
.venv/bin/streamlit run streamlit_app.py
```

Puis ouvrir :

```text
http://127.0.0.1:8501
```

L'application propose trois modes :

- `keyword` : recherche BM25 sur index inversé.
- `vector` : embeddings `BAAI/bge-m3` avec Chroma.
- `hybrid` : fusion RRF entre les résultats BM25 et vectoriels.

## Génération Gemini optionnelle

La recherche locale fonctionne sans clé API. Pour générer une réponse avec Gemini, définir :

```bash
GEMINI_API_KEY=your_key_here
```

ou créer un fichier local `.env` :

```text
GEMINI_API_KEY=your_key_here
```

La génération est implémentée dans `p4_rag_generate.py`. Les règles de génération visent à limiter les hallucinations :

- réponse fondée uniquement sur le contexte retrouvé ;
- refus d'inventer des ingrédients, quantités, étapes ou sources ;
- indication explicite lorsque le contexte ne suffit pas ;
- détection des requêtes adversariales ;
- neutralisation des instructions hostiles présentes dans les documents retrouvés.

## Utilisation en ligne de commande

Recherche par mots-clés :

```powershell
.\.venv\Scripts\python.exe p2_keyword_retrieval.py --query "risotto champignons parmesan" --top-k 5
```

Recherche hybride :

```powershell
.\.venv\Scripts\python.exe p3_hybrid_retrieval.py --query "recette ratatouille" --mode hybrid --top-k 5
```

Génération RAG :

```powershell
.\.venv\Scripts\python.exe p4_rag_generate.py --query "Je voudrais faire une ratatouille." --retrieval-mode hybrid --top-k 5
```

Interface web locale sans Streamlit :

```powershell
.\.venv\Scripts\python.exe web_app.py --host 127.0.0.1 --port 8000
```

Puis ouvrir :

```text
http://127.0.0.1:8000
```

## Évaluation

Évaluation de la recherche par mots-clés :

```powershell
.\.venv\Scripts\python.exe p2_keyword_retrieval.py --eval data\keyword_eval_queries.jsonl --mode ranked
```

Comparaison des modes `keyword`, `vector` et `hybrid` :

```powershell
.\.venv\Scripts\python.exe p3_hybrid_retrieval.py --eval data\keyword_eval_queries.jsonl --compare
```

Évaluation RAG et sécurité :

```powershell
.\.venv\Scripts\python.exe p5_rag_evaluate.py --mode hybrid --top-k 5
.\.venv\Scripts\python.exe p5_rag_evaluate.py --security-only
```

Résultats vérifiés localement :

| Mode | Recall@1 | Precision@1 | MRR |
|---|---:|---:|---:|
| keyword | 1.000 | 1.000 | 1.000 |
| vector | 0.875 | 0.875 | 0.938 |
| hybrid | 1.000 | 1.000 | 1.000 |

Évaluation sécurité :

```text
score = 1.0
```

Les tests couvrent les prompt injections, les tentatives de fuite du prompt, les jailbreaks et la sanitisation du contexte hostile.

## Reproductibilité

Points à vérifier lors d'une installation sur une nouvelle machine :

- `chroma_db/` doit exister ou être reconstruit.
- `.hf_cache/` doit contenir le modèle `BAAI/bge-m3` si le chargement local est utilisé.
- Si le cache HuggingFace est absent, il faut préparer le modèle avec un accès réseau.
- La génération Gemini nécessite `GEMINI_API_KEY`.

Construire la base Chroma :

```powershell
.\.venv\Scripts\python.exe p3_hybrid_retrieval.py --build-vectorstore --jsonl data\chunks.jsonl
```

## Correspondance avec les consignes

| Consigne | Réalisation |
|---|---|
| Sujet et données en français | Corpus de recettes françaises dans `data/recipes.jsonl` |
| RAG | `p2_keyword_retrieval.py`, `p3_hybrid_retrieval.py`, `p4_rag_generate.py` |
| Évaluation du RAG | `p5_rag_evaluate.py`, `data/keyword_eval_queries.jsonl` |
| Gestion des hallucinations | prompt contraint, faithfulness, refus de génération non fondée |
| Application Streamlit | `streamlit_app.py` |
| Code source reproductible | `requirements.txt`, scripts CLI, données prétraitées |

## Rendus à préparer pour le cours

Les consignes demandent aussi :

- un rapport de 8 pages maximum avec le pourcentage de contribution ;
- une présentation et une démo ;
- un déploiement éventuel en bonus.

Ces éléments doivent être préparés séparément pour le rendu final.

## Licence

Voir [LICENSE](LICENSE).
