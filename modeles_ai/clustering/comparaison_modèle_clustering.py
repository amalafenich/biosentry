from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from sklearn.decomposition import PCA

import hdbscan
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONNEXION MONGODB
# =========================================================
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["biosentry_db"]

# =========================================================
# 2. CHARGEMENT DES DONNÉES
# =========================================================
print("=== CHARGEMENT DATA_AGGREGATION ===")

df = pd.DataFrame(
    db["data_aggregation"].find({}, {"_id": 0})
)

if df.empty:
    print("❌ Aucune donnée trouvée.")
    exit()

print(f"Documents chargés : {len(df)}")

# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================
df["mention_count"] = pd.to_numeric(
    df["mention_count"],
    errors="coerce"
).fillna(0)

df["year"]  = df["period"].str[:4].astype(int)
df["month"] = df["period"].str[5:7].astype(int)

stats = (
    df.groupby(["drug", "side_effect"])["mention_count"]
    .agg(
        mean_count="mean",
        std_count="std",
        total_mentions="sum",
        num_periods="count",
        max_count="max",
        min_count="min"
    )
    .fillna(0)
    .reset_index()
)

# =========================================================
# 4. COEFFICIENT DE VARIATION
# =========================================================
stats["cv"] = np.where(
    stats["mean_count"] > 0,
    stats["std_count"] /
    stats["mean_count"].clip(lower=1),
    0
)

# =========================================================
# 5. AJOUT YEAR_LAST / MONTH_LAST
# =========================================================
last_period = (
    df.sort_values("period")
    .groupby(["drug", "side_effect"])[["year", "month"]]
    .last()
    .reset_index()
    .rename(columns={
        "year":  "year_last",
        "month": "month_last"
    })
)

stats = stats.merge(
    last_period,
    on=["drug", "side_effect"],
    how="left"
)

# =========================================================
# 6. FEATURES
# =========================================================
features = [
    "mean_count",
    "std_count",
    "total_mentions",
    "num_periods",
    "max_count",
    "min_count",
    "cv",
    "year_last",
    "month_last"
]

X = stats[features].fillna(0)

print(f"\nShape matrice : {X.shape}")

# =========================================================
# 7. NORMALISATION
# =========================================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# 8. PCA  (pour HDBSCAN uniquement)
# =========================================================
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

print(f"Dimensions après PCA : {X_pca.shape[1]}")

# =========================================================
# 9. FONCTION D'ÉVALUATION
# =========================================================
def evaluate_clustering(X, labels, model_name):

    n_clusters = (
        len(set(labels))
        - (1 if -1 in labels else 0)
    )

    n_outliers = int((labels == -1).sum())

    if -1 in labels:
        mask         = labels != -1
        X_eval       = X[mask]
        labels_eval  = labels[mask]
    else:
        X_eval       = X
        labels_eval  = labels

    if len(set(labels_eval)) < 2:
        silhouette = -1
        calinski   = -1
        davies     = -1
    else:
        silhouette = silhouette_score(X_eval, labels_eval)
        calinski   = calinski_harabasz_score(X_eval, labels_eval)
        davies     = davies_bouldin_score(X_eval, labels_eval)

    return {
        "Model":             model_name,
        "Clusters":          n_clusters,
        "Outliers":          n_outliers,
        "Silhouette":        round(silhouette, 4),
        "Calinski_Harabasz": round(calinski, 2),
        "Davies_Bouldin":    round(davies, 4)
    }

# =========================================================
# 10. KMEANS
# =========================================================
print("\n=== KMEANS ===")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_results = evaluate_clustering(
    X_scaled,
    kmeans_labels,
    "KMeans"
)

# =========================================================
# 11. AGGLOMERATIVE
# =========================================================
print("\n=== AGGLOMERATIVE ===")

agglo = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

agglo_labels = agglo.fit_predict(X_scaled)

agglo_results = evaluate_clustering(
    X_scaled,
    agglo_labels,
    "Agglomerative"
)

# =========================================================
# 12. HDBSCAN  (paramètres optimaux issus du scan)
# =========================================================
print("\n=== HDBSCAN ===")

hdb = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=20,
    metric="euclidean"
)

hdb_labels = hdb.fit_predict(X_pca)

hdb_results = evaluate_clustering(
    X_pca,
    hdb_labels,
    "HDBSCAN"
)

# =========================================================
# 13. TABLEAU COMPARATIF
# =========================================================
results_df = pd.DataFrame([
    kmeans_results,
    agglo_results,
    hdb_results
])

print("\n==============================")
print("📊 COMPARAISON DES MODÈLES")
print("==============================")

print(results_df.to_string(index=False))

# =========================================================
# 14. INTERPRÉTATION AUTOMATIQUE
# =========================================================
print("\n==============================")
print("📌 INTERPRÉTATION")
print("==============================")

best_silhouette = results_df.loc[
    results_df["Silhouette"].idxmax()
]

best_calinski = results_df.loc[
    results_df["Calinski_Harabasz"].idxmax()
]

best_davies = results_df.loc[
    results_df["Davies_Bouldin"].idxmin()
]

print(
    f"\n✅ Meilleur Silhouette Score  : "
    f"{best_silhouette['Model']} "
    f"({best_silhouette['Silhouette']})"
)

print(
    f"✅ Meilleur Calinski-Harabasz : "
    f"{best_calinski['Model']} "
    f"({best_calinski['Calinski_Harabasz']})"
)

print(
    f"✅ Meilleur Davies-Bouldin    : "
    f"{best_davies['Model']} "
    f"({best_davies['Davies_Bouldin']})"
)

# =========================================================
# 15. VOTE FINAL
# =========================================================
print("\n==============================")
print("🏆 MODÈLE LE PLUS EFFICACE")
print("==============================")

votes = {model: 0 for model in results_df["Model"]}

votes[best_silhouette["Model"]] += 1
votes[best_calinski["Model"]]   += 1
votes[best_davies["Model"]]     += 1

best_model_name = max(votes, key=votes.get)

print(f"\n🏆 Meilleur modèle global : {best_model_name}")
print(f"   Votes : {votes}")

# =========================================================
# 16. EXPLICATION DES MÉTRIQUES
# =========================================================
print("\n==============================")
print("📖 SIGNIFICATION DES MÉTRIQUES")
print("==============================")

print("""
1. Silhouette Score
   → Entre -1 et 1. Plus proche de 1 = meilleur clustering

2. Calinski-Harabasz
   → Plus grand = clusters mieux séparés et compacts

3. Davies-Bouldin
   → Plus petit = meilleur clustering (idéal proche de 0)

4. Outliers
   → Nombre de points non assignés à un cluster
   → Pertinent surtout pour HDBSCAN
""")

# =========================================================
# 17. FERMETURE
# =========================================================
client.close()

print("\n✅ Comparaison terminée.")