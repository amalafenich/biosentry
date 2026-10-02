# BioSentry

**Plateforme de pharmacovigilance pour la détection précoce des effets secondaires de médicaments.**

BioSentry collecte des données issues de sources médicales publiques, extrait et normalise les effets secondaires mentionnés, les agrège par couple *(médicament, effet secondaire, période)*, puis applique des modèles de clustering et de détection d'anomalies pour faire remonter les signaux faibles et attribuer un niveau d'alerte.

> **Équipe :** Aya Chakour · Amal Afenich · Oumaima Enajjari · Khouloud El Hajji

---

## Sommaire

1. [Contexte et problématique](#contexte-et-problématique)
2. [Objectifs](#objectifs)
3. [Architecture](#architecture)
4. [Structure du dépôt](#structure-du-dépôt)
5. [Sources de données](#sources-de-données)
6. [Schéma des données](#schéma-des-données)
7. [Modèles](#modèles)
8. [Installation](#installation)
9. [Utilisation](#utilisation)
10. [Dashboard](#dashboard)
11. [Livrables](#livrables)
12. [Limites et avertissements](#limites-et-avertissements)
13. [Technologies](#technologies)

---

## Contexte et problématique

Les essais cliniques menés avant la mise sur le marché d'un médicament portent sur un nombre limité de patients et sur une durée courte : certains effets secondaires rares ou tardifs n'apparaissent qu'une fois le médicament largement utilisé. La pharmacovigilance consiste à surveiller ces effets après commercialisation, mais les informations sont dispersées entre de multiples sources (fiches officielles, publications scientifiques, essais cliniques, bases de notifications, forums de patients) et leur analyse manuelle est lente.

BioSentry répond à ce problème en regroupant ces sources dans une base unique, en suivant l'évolution des mentions d'effets secondaires dans le temps, puis en signalant automatiquement les évolutions anormales grâce à plusieurs algorithmes complémentaires. L'objectif est d'**aider à repérer plus tôt un signal à investiguer**, et non de remplacer l'expertise d'un professionnel de santé.

## Objectifs

- Centraliser des données hétérogènes sur les effets indésirables des médicaments
- Extraire et normaliser les entités médicales (médicaments, effets, symptômes, dosages) par NLP biomédical
- Pondérer la fiabilité des sources (score d'autorité de type HITS)
- Identifier des groupes de signaux similaires (clustering)
- Détecter les pics de mentions anormaux dans le temps (anomalies)
- Produire un niveau d'alerte exploitable : `NORMAL`, `MODEREE`, `ELEVEE`, `CRITIQUE`

## Architecture

```
 Sources : MedlinePlus, PubMed, ClinicalTrials.gov, openFDA,
           Drugs.com, WebMD, Reddit
                         |
                         v
        Liste des médicaments (RxNorm) ──> drug_list
                         |
                         v
               Scrapers / API (collecte)
                         |
                         v
   NLP (BioBERT) + score d'autorité (HITS, Neo4j)
                         |
                         v
   Normalisation + extraction des dates ──> normalised_sources
                         |
                         v
             Contrôle qualité (quality_flags)
                         |
                         v
 MongoDB Atlas (biosentry_db)
   drug_list -> data_aggregation -> ml_predictions
                         |
                         v
 Modèles IA
   - clustering : K-Means, agglomératif, HDBSCAN
   - anomalies  : Z-score glissant, Isolation Forest, LOF
                         |
                         v
        Niveau d'alerte (alerte.py) -> Dashboard
```

## Structure du dépôt

```
biosentry/
├── collecte_des_donnees/
│   ├── 01.Rxnorm/
│   │   └── rxnorm_builder.py                   # Construit drug_list via l'API RxNorm (par classe ATC)
│   ├── 02.Scrapers/
│   │   ├── biosentry_medlineplus_scraper_full.py   # MedlinePlus (Selenium + BeautifulSoup)
│   │   ├── scraper_pubmed.py                       # PubMed (API NCBI E-utilities)
│   │   ├── scraper_clinicaltrials.py               # ClinicalTrials.gov (API v2)
│   │   ├── scraper_openfda.py                      # openFDA (rapports d'événements indésirables)
│   │   ├── scraper_drugscom.py                     # Drugs.com (Selenium)
│   │   ├── scraper_webmd.py                        # WebMD (Selenium)
│   │   └── scraper_reddit.py                       # Reddit (publications de patients)
│   ├── 03.NLP/
│   │   └── nlp_biobert.py                      # NER et classification biomédicales (BioBERT, VADER)
│   ├── 04.Hits/
│   │   └── hits_authority.py                   # Score d'autorité des sources (HITS, Neo4j)
│   ├── 05.Normalisation_et_Dates/
│   │   ├── biosentry_normalisation.py          # Nettoyage et normalisation des termes
│   │   ├── Date_Extraction.py                  # Extraction et harmonisation des dates
│   │   └── merge_normalized_with_dates.py      # Fusion données normalisées + dates
│   ├── 06.database_mongodb/
│   │   ├── data_initial.json                   # Données avant traitement
│   │   ├── data_final.json                     # Données après traitement
│   │   └── drug_list.json                      # Liste des médicaments
│   └── 07.Data_quality/
│       └── quality_flags.py                    # Drapeaux de qualité des données
│
├── modeles_ai/
│   ├── clustering/
│   │   ├── kmeans_clustering.py
│   │   ├── agglomeratif.py
│   │   ├── hdbscan.py
│   │   ├── comparaison_modele_clustering.py    # Comparaison des 3 modèles
│   │   ├── fusion_final.py                     # Fusion des résultats des 3 modèles
│   │   └── *.json                              # Résultats de clustering
│   └── anomalie_detection/
│       ├── zscore_rolling.py
│       ├── iforest.py
│       ├── anomalies_lof.py
│       ├── alerte.py                           # Calcul du niveau d'alerte
│       └── *.json                              # Résultats de détection
│
│  
├── Presentation_BioSentry.pdf
├── Rapport_de_projet.pdf
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Sources de données

| Source | Méthode | Contenu |
|---|---|---|
| RxNorm | API REST NLM | Liste de médicaments par classe thérapeutique (ATC) |
| MedlinePlus | Scraping (Selenium + BeautifulSoup) | Fiches médicaments, classes, noms commerciaux, effets secondaires, avertissements |
| PubMed | API NCBI E-utilities | Articles sur les effets indésirables (2019 à 2026) |
| ClinicalTrials.gov | API v2 | Essais cliniques liés aux effets indésirables (2019 à 2026) |
| openFDA | API `drug/event` | Rapports d'événements indésirables |
| Drugs.com | Scraping (Selenium) | Avis et effets secondaires rapportés |
| WebMD | Scraping (Selenium) | Avis et effets secondaires rapportés |
| Reddit | Requêtes HTTP sur des subreddits ciblés | Témoignages de patients |

> **Attention :** les monographies MedlinePlus sont protégées par un copyright (ASHP). Leur usage dans ce projet est strictement limité à la **recherche académique**. Les conditions d'utilisation de chaque site scrapé doivent également être respectées.

## Schéma des données

### Données brutes MedlinePlus

Un document par médicament :

```json
{
  "url": "https://medlineplus.gov/druginfo/meds/a606008.html",
  "drug_name": "Phenylephrine",
  "drug_class": "nasal decongestants",
  "brand_names": ["Lusonal", "Sudafed PE Congestion"],
  "prescribed_for": ["Phenylephrine is used to relieve nasal discomfort ..."],
  "side_effects": {
    "mild": [],
    "serious": ["nervousness", "dizziness", "sleeplessness"],
    "total": 3
  },
  "warnings": [],
  "source": "MedlinePlus",
  "scraped_at": "2026-05-16 18:37:18"
}
```

| Champ | Type | Description |
|---|---|---|
| `url` | texte | Page MedlinePlus d'origine |
| `drug_name` | texte | Nom du médicament |
| `drug_class` | texte | Classe thérapeutique |
| `brand_names` | liste de textes | Noms commerciaux |
| `prescribed_for` | liste de textes | Indications du médicament |
| `side_effects.mild` | liste de textes | Effets secondaires légers |
| `side_effects.serious` | liste de textes | Effets secondaires graves |
| `side_effects.total` | entier | Nombre total d'effets secondaires |
| `warnings` | liste de textes | Avertissements |
| `source` | texte | Source de la donnée |
| `scraped_at` | texte | Date et heure de collecte |

### Collections MongoDB (base `biosentry_db`)

| Collection | Champs principaux | Rôle |
|---|---|---|
| `drug_list` | `name`, `category` | Liste des médicaments à interroger (PubMed, ClinicalTrials, etc.) |
| `normalised_sources` | texte normalisé, entités extraites, dates, score d'autorité | Données nettoyées issues de toutes les sources |
| `data_aggregation` | `drug`, `side_effect`, `period` (AAAA-MM), `mention_count` | Nombre de mentions par médicament, effet et mois. **Entrée des modèles** |
| `ml_predictions` | `drug`, `side_effect`, `period`, `lof_anomaly`, `iforest_anomaly`, `zscore_anomaly`, `nb_algos`, `alert_level` | Résultats des détecteurs d'anomalies et niveau d'alerte final |

## Modèles

### Pré-traitement NLP

- **Extraction d'entités** (médicaments, effets secondaires, symptômes, dosages) avec un modèle BioBERT, complétée par un classifieur et une analyse de sentiment (VADER).
- **Score d'autorité** des sources calculé avec l'algorithme HITS (graphe stocké dans Neo4j), afin de pondérer la fiabilité de chaque source.
- **Normalisation** des termes et **extraction des dates** (période 2019 à 2026).

### Clustering

Appliqué sur les données agrégées (`data_aggregation`) avec des variables normalisées (`StandardScaler`).

- **K-Means**
- **Clustering agglomératif**
- **HDBSCAN** : distingue les clusters de signal, les points aberrants (drapeaux critiques en pharmacovigilance) et le bruit

`comparaison_modele_clustering.py` compare les modèles avec les métriques *silhouette*, *Calinski-Harabasz* et *Davies-Bouldin* (visualisation par PCA).

### Détection d'anomalies

Variables utilisées : nombre de mentions, moyenne, écart-type, total de mentions, nombre de périodes, mois.

- **Z-score glissant** (fenêtre adaptée à la longueur de chaque série)
- **Isolation Forest**
- **Local Outlier Factor (LOF)**

### Niveau d'alerte

`alerte.py` compte combien d'algorithmes signalent une anomalie pour un même triplet *(médicament, effet, période)* :

| Algorithmes en alerte | Niveau |
|:---:|---|
| 3 | `CRITIQUE` |
| 2 | `ELEVEE` |
| 1 | `MODEREE` |
| 0 | `NORMAL` |

## Installation

**Prérequis :** Python 3.10+, un cluster MongoDB (Atlas ou local), Google Chrome (scrapers Selenium) et une instance Neo4j locale (uniquement pour le calcul HITS).

```bash
git clone https://github.com/amalafenich/biosentry.git
cd biosentry
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # modèle spaCy utilisé par la normalisation (à adapter si un autre modèle est chargé)
```

**Configuration :** copier `.env.example` en `.env` et renseigner les valeurs.

```env
MONGO_URI=mongodb+srv://utilisateur:motdepasse@cluster.mongodb.net/
NCBI_API_KEY=votre_cle_ncbi
NCBI_EMAIL=votre_email@exemple.com
OPENFDA_API_KEY=votre_cle_openfda
NEO4J_PASSWORD=votre_mot_de_passe_neo4j
```

> **Ne jamais versionner le fichier `.env`** (il est exclu par le `.gitignore`).

## Utilisation

Ordre d'exécution recommandé :

```bash
# 1. Liste des médicaments (crée la collection drug_list)
python collecte_des_donnees/01.Rxnorm/rxnorm_builder.py

# 2. Collecte
python collecte_des_donnees/02.Scrapers/biosentry_medlineplus_scraper_full.py
python collecte_des_donnees/02.Scrapers/scraper_pubmed.py
python collecte_des_donnees/02.Scrapers/scraper_clinicaltrials.py
python collecte_des_donnees/02.Scrapers/scraper_openfda.py
python collecte_des_donnees/02.Scrapers/scraper_drugscom.py
python collecte_des_donnees/02.Scrapers/scraper_webmd.py
python collecte_des_donnees/02.Scrapers/scraper_reddit.py

# 3. NLP, autorité des sources, normalisation et dates
#    (biosentry_normalisation.py importe nlp_biobert et hits_authority :
#     les trois modules doivent être accessibles depuis le PYTHONPATH)
python collecte_des_donnees/05.Normalisation_et_Dates/biosentry_normalisation.py
python collecte_des_donnees/05.Normalisation_et_Dates/Date_Extraction.py
python collecte_des_donnees/05.Normalisation_et_Dates/merge_normalized_with_dates.py

# 4. Contrôle qualité
python collecte_des_donnees/07.Data_quality/quality_flags.py

# 5. Clustering (depuis le dossier concerné)
cd modeles_ai/clustering
python kmeans_clustering.py
python agglomeratif.py
python hdbscan.py
python comparaison_modele_clustering.py
python fusion_final.py
cd ../..

# 6. Détection d'anomalies
cd modeles_ai/anomalie_detection
python zscore_rolling.py
python iforest.py
python anomalies_lof.py
python alerte.py
```

**Remarques :**

- Les scrapers lisent la liste des médicaments dans la collection `drug_list`, qui doit exister avant leur lancement.
- Les scripts de modèles lisent la collection `data_aggregation`, et `alerte.py` lit `ml_predictions`.
- `fusion_final.py` attend un fichier nommé `agglo_results.json` : renommer `agglomeratif_results.json` si nécessaire.
- Les fichiers `data_initial.json` et `data_final.json` permettent de consulter les données sans relancer toute la collecte.

## Dashboard

Une vidéo de démonstration du tableau de bord est disponible ici :
**[Voir la vidéo de démonstration](https://drive.google.com/drive/u/1/folders/1ntOrtv8fSR99Z7nLgu1l801PlHqUaTGy)**

## Livrables

- Présentation du projet : [`Presentation_BioSentry.pdf`](Presentation_BioSentry.pdf)
- Rapport de projet : [`Rapport_de_projet.pdf`](Rapport_de_projet.pdf)
- Code de collecte, de traitement et de modélisation
- Données collectées et résultats des modèles au format JSON
- Vidéo de démonstration du dashboard (lien ci-dessus)


## Technologies

Python, pandas, NumPy, scikit-learn, HDBSCAN, PyMongo, Neo4j, Selenium, BeautifulSoup, requests, lxml, spaCy, PyTorch, Hugging Face Transformers (BioBERT), VADER, python-dotenv, tqdm.
