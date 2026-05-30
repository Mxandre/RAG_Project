# RAG_LO17 — Projet RAG sur recettes françaises

[中文](README.md) · [Français](README.fr.md)

Projet de génération augmentée par récupération (RAG) construit sur un corpus de recettes françaises issues de [jemangefrancais.com](https://jemangefrancais.com). Ce dépôt ne contient pour l'instant que l'étape **P1 — Ingestion des données** : collecte, parsing, nettoyage et découpage en chunks. L'aval consomme l'interface stable `run() -> list[Chunk]` et alimente un récupérateur hybride (vecteur + BM25).

## Environnement et commandes

Python 3.x requis. Le Python système est protégé par PEP-668 (externally-managed) ; l'installation doit passer par un venv local.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python p1_ingestion.py                 # crawl complet de toutes les pages de catégorie
.venv/bin/python p1_ingestion.py --max-pages 1   # debug : première page seulement
.venv/bin/python p1_ingestion.py --delay 0.4 --max-chars 800
```

Options :
- `--max-pages` : limite de pagination (debug)
- `--delay` : intervalle entre requêtes en secondes (politesse envers le site source)
- `--max-chars` : seuil de découpage des étapes

## Architecture du pipeline

Trois étapes, tous les artefacts dans `data/`, ré-exécutables et idempotents :

```
crawl  → data/raw/<slug>.html         (collecte d'URLs → téléchargement, skip si présent)
parse  → data/recipes.jsonl           (parse_recipe route vers trois stratégies)
chunk  → data/chunks.jsonl + List[Chunk]
```

`run()` orchestre les trois étapes. `data/raw/` est gitignored ; les deux fichiers `.jsonl` sont commités comme artefacts partagés.

## Stratégie de parsing

Le dispatcher `parse_recipe()` classe le HTML puis route vers la stratégie correspondante :

- **multi** — pages contenant ≥2 sous-titres `<h3>` numérotés correspondant à des recettes indépendantes (ex. *Les délices du Risotto* avec 3 recettes distinctes). Chaque sous-recette est isolée et parsée séparément, émise comme enregistrement `slug#N`.
- **standard** — pages mono-recette en disposition courante. Parcours en un seul passage dans l'ordre du document, avec les règles suivantes :
  - h2/h3/h4/strong/b ainsi que les lignes `<p>` ancrées en début → marqueurs de section
  - Sous-titres courts terminés par `:` (ex. `Le brownie :`) → étiquettes de sous-groupes d'ingrédients
  - Les sous-titres courts contenant `recette` / `façon` basculent vers la section étapes (motif girolles)
  - Marqueurs d'étapes étendus aux ordinaux numérotés `1ère étape` / `2ème étape` (motif cèpe)
  - Une grappe de `<li>` sans en-tête de section explicite est rattachée par défaut aux ingrédients ; à l'intérieur d'`ingredients`, un `<p>` en prose sans tiret ouvre implicitement la section étapes
- **skip** — absence de conteneur / aucun marqueur explicite ingrédient/préparation/étape / aucune section étapes récupérée → consigné dans `data/unparsed.md` avec la raison

Les listes d'étapes sont fusionnées en un seul paragraphe avec `" ".join` avant persistance, ce qui améliore le rappel vectoriel sur les pages aux étapes très fragmentées (ex. *entremet* avec plus de 30 micro-`<li>`).

Détails de couverture et statistiques dans [`docs/p1_parsing_report.md`](docs/p1_parsing_report.md) ; liste des fichiers non parsés dans [`data/unparsed.md`](data/unparsed.md).

## Contrat de données

`Chunk` est l'interface stable que P1 expose aux étages aval :

```python
@dataclass
class Chunk:
    id: str          # f"{slug}::{section_type}", suffixé ::{idx} si plusieurs blocs d'étapes
    text: str        # corps de récupération : indexé par BM25 et embeddé pour vecteurs
    metadata: dict   # {recipe_name, source_url, section_type, [servings, difficulty]}
```

Découpage par champ, pas par fenêtre fixe : une recette produit un chunk `ingredients` + un chunk `steps` (redécoupé sur frontières d'étapes si `max_chars` est dépassé, avec `chunk_index`). Le champ `section_type` permet aux étages aval de pondérer différemment (ingrédients favorisent les correspondances BM25 par mot-clé ; étapes favorisent la sémantique vectorielle).

`recipes.jsonl` conserve en plus un dictionnaire `meta` (`prep_time` / `cook_time` / `rest_time` / `difficulty` / `servings`), exclu du corps de récupération.

## Vérification

Pas de framework de tests. Régressions vérifiées par exécution sur un petit échantillon `--max-pages` + assertions `jq` :

```bash
.venv/bin/python p1_ingestion.py --max-pages 1
jq -r '.slug' data/recipes.jsonl | sort > /tmp/cur.txt
wc -l data/recipes.jsonl                                   # ≥112
jq 'select(.steps|length==0)' data/recipes.jsonl           # doit être vide (records ingrédients-seuls rejetés)
```

Après modification du parseur, relancer le pipeline complet et diff'er l'ensemble des slugs dans `recipes.jsonl` ; chaque slug auparavant parsable doit toujours apparaître (le slug racine Risotto remplacé par trois sous-records `slug#N` est attendu).

## Structure du dépôt

```
p1_ingestion.py            # tout le code P1 (crawl/parse/chunk + génération de rapports)
requirements.txt
data/
  raw/                     # .gitignore : HTML téléchargés
  recipes.jsonl            # versionné : recettes parsées
  chunks.jsonl             # versionné : chunks produits
  unparsed.md              # versionné : fichiers ignorés avec raisons
docs/
  p1_ingestion_report.md   # évolution du parseur et inventaire structurel
  p1_parsing_report.md     # couverture courante et description de la stratégie
CLAUDE.md                  # consignes projet pour Claude Code (contraintes + contexte)
```

## Licence

Voir [LICENSE](LICENSE).
