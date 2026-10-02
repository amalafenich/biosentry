from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import hdbscan
import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# 1. CONNEXION MONGODB
# =====================================================
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["biosentry_db"]

# =====================================================
# 2. CHARGEMENT
# =====================================================
print("=== CHARGEMENT DATA_AGGREGATION ===")

df = pd.DataFrame(
    db["data_aggregation"].find({}, {"_id": 0})
)

if df.empty:
    print("❌ Aucune donnée trouvée.")
    exit()

df["mention_count"] = pd.to_numeric(
    df["mention_count"],
    errors="coerce"
).fillna(0)

print(f"Documents chargés : {len(df)}")

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================
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

stats["cv"] = np.where(
    stats["mean_count"] > 0,
    stats["std_count"] / stats["mean_count"].clip(lower=1),
    0
)

last_period = (
    df.sort_values("period")
    .groupby(["drug", "side_effect"])[["year", "month"]]
    .last()
    .reset_index()
    .rename(columns={"year": "year_last", "month": "month_last"})
)

stats = stats.merge(last_period, on=["drug", "side_effect"], how="left")

print(f"Couples totaux : {len(stats)}")

# =====================================================
# 4. FILTRAGE BRUIT / SIGNAL
# =====================================================
bruit  = stats[stats["total_mentions"] <= 1].copy()
signal = stats[stats["total_mentions"] >= 2].copy()

print(f"Couples bruit  (1 mention)   : {len(bruit)}")
print(f"Couples signal (>=2 mentions): {len(signal)}")

if len(signal) == 0:
    print("❌ Aucun signal exploitable.")
    exit()

# =====================================================
# 5. FEATURES ET NORMALISATION
# =====================================================
feature_cols = [
    "mean_count",
    "std_count",
    "num_periods",
    "max_count",
    "min_count",
    "cv",
    "year_last",
    "month_last"
]

X_scaled = StandardScaler().fit_transform(
    signal[feature_cols]
)

print(f"Shape matrice : {X_scaled.shape}")

# =====================================================
# 6. PCA
# =====================================================
pca   = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Dimensions après PCA : {X_pca.shape[1]}")

# =====================================================
# 7. HDBSCAN  (paramètres optimaux)
# =====================================================
print("\n=== HDBSCAN ===")

model = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=20,
    metric="euclidean"
)

labels = model.fit_predict(X_pca)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = int((labels == -1).sum())

print(f"Clusters trouvés : {n_clusters}")
print(f"Outliers         : {n_outliers}")

# =====================================================
# 8. APPLICATION SUR SIGNAL
# =====================================================
signal = signal.copy()
signal["hdbscan_cluster"] = labels

# =====================================================
# 9. BRUIT MÉTIER  → cluster = -2
# =====================================================
bruit = bruit.copy()
bruit["hdbscan_cluster"] = -2

# =====================================================
# 10. CONCAT SIGNAL + BRUIT
# =====================================================
full = pd.concat([signal, bruit], ignore_index=True)

print("\nDistribution des clusters :")
print(
    full["hdbscan_cluster"]
    .value_counts()
    .sort_index()
    .to_string()
)

# =====================================================
# 11. EXPORT JSON  (drug, side_effect, period, cluster)
# =====================================================

# Rejoindre avec df original pour avoir la colonne period
df_result = df[["drug", "side_effect", "period"]].drop_duplicates()

df_result = df_result.merge(
    full[["drug", "side_effect", "hdbscan_cluster"]],
    on=["drug", "side_effect"],
    how="left"
)

# Valeur par défaut si non trouvé
df_result["hdbscan_cluster"] = df_result["hdbscan_cluster"].fillna(-2).astype(int)

output = []

for _, row in df_result.iterrows():
    output.append({
        "drug":            row["drug"],
        "side_effect":     row["side_effect"],
        "period":          row["period"],
        "hdbscan_cluster": int(row["hdbscan_cluster"])
    })

with open("hdbscan_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\n✅ JSON créé : hdbscan_results.json ({len(output)} lignes)")

# =====================================================
# 12. SAUVEGARDE MONGODB
# =====================================================
db.hdbscan_clusters.drop()
db.hdbscan_clusters.insert_many(output)

print("✅ Sauvegardé dans MongoDB : collection hdbscan_clusters")

client.close()
print("\n✅ Pipeline HDBSCAN terminé.")
