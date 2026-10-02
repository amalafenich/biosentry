from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import os
from datetime import datetime

# ── Chargement MongoDB ────────────────────────────────────────────
load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))["biosentry_db"]

print("=== CHARGEMENT DATA_AGGREGATION ===")
df = pd.DataFrame(db["data_aggregation"].find({}, {"_id": 0}))
print(f"Documents chargés : {len(df)}")

# ── Features engineering ──────────────────────────────────────────
df["year"]  = df["period"].str[:4].astype(int)
df["month"] = df["period"].str[5:7].astype(int)

stats = df.groupby(["drug", "side_effect"])["mention_count"].agg(
    mean_count="mean",
    std_count="std",
    total_mentions="sum",
    num_periods="count"
).reset_index()

stats["std_count"] = stats["std_count"].fillna(0)

df = df.merge(stats, on=["drug", "side_effect"], how="left")

# ── Features utilisées par KMeans ─────────────────────────────────
features = [
    "mention_count",
    "mean_count",
    "std_count",
    "total_mentions",
    "num_periods",
    "month"
]

X = df[features].fillna(0)

# ── Normalisation ─────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── KMeans ────────────────────────────────────────────────────────
print("Calcul KMeans en cours...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# ── Résumé ────────────────────────────────────────────────────────
print(f"\nDistribution des clusters :")
print(df["kmeans_cluster"].value_counts().sort_index().to_string())

print("\nTop 10 exemples :")
print(
    df[["drug", "side_effect", "period", "mention_count", "kmeans_cluster"]]
    .head(10)
    .to_string(index=False)
)

# ── JSON propre ───────────────────────────────────────────────────
output = []

for _, row in df.iterrows():
    output.append({
        "algorithm": "KMeans",
        "drug": row["drug"],
        "side_effect": row["side_effect"],
        "period": row["period"],
        "mention_count": int(row["mention_count"]),
        "kmeans_cluster": int(row["kmeans_cluster"]),
        "created_at": datetime.utcnow().isoformat()
    })

# ── Sauvegarde JSON ───────────────────────────────────────────────
output_file = "kmeans_results.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\n✅ JSON créé : {output_file}")
print(f"Total documents sauvegardés : {len(output)}")