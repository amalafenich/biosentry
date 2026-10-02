from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
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
# 4. FEATURES ET NORMALISATION
# =====================================================
feature_cols = [
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

X_scaled = StandardScaler().fit_transform(
    stats[feature_cols].fillna(0)
)

print(f"Shape matrice : {X_scaled.shape}")

# =====================================================
# 5. AGGLOMERATIVE
# =====================================================
print("\n=== AGGLOMERATIVE ===")

agglo = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

stats["agglo_cluster"] = agglo.fit_predict(X_scaled)

print("Distribution des clusters :")
print(
    stats["agglo_cluster"]
    .value_counts()
    .sort_index()
    .to_string()
)

# =====================================================
# 6. EXPORT JSON  (drug, side_effect, period, cluster)
# =====================================================

# Rejoindre avec df original pour avoir la colonne period
df_result = df[["drug", "side_effect", "period"]].drop_duplicates()

df_result = df_result.merge(
    stats[["drug", "side_effect", "agglo_cluster"]],
    on=["drug", "side_effect"],
    how="left"
)

output = []

for _, row in df_result.iterrows():
    output.append({
        "drug":          row["drug"],
        "side_effect":   row["side_effect"],
        "period":        row["period"],
        "agglo_cluster": int(row["agglo_cluster"])
    })

with open("agglo_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\n✅ JSON créé : agglo_results.json ({len(output)} lignes)")

# =====================================================
# 7. SAUVEGARDE MONGODB
# =====================================================
db.agglo_clusters.drop()
db.agglo_clusters.insert_many(output)

print("✅ Sauvegardé dans MongoDB : collection agglo_clusters")

client.close()
print("\n✅ Pipeline Agglomerative terminé.")
