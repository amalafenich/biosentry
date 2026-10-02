from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.neighbors import LocalOutlierFactor
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

# ── Features utilisées par LOF ────────────────────────────────────
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

# ── LOF ───────────────────────────────────────────────────────────
print("Calcul LOF en cours...")

lof = LocalOutlierFactor(
    n_neighbors=30,
    contamination="auto"
)

labels = lof.fit_predict(X_scaled)
lof_scores = -lof.negative_outlier_factor_

df["lof_score"] = lof_scores
df["is_anomaly"] = (labels == -1)

# ── Extraction anomalies uniquement ───────────────────────────────
anomalies = df[df["is_anomaly"] == True].copy()

print(f"\nTotal anomalies détectées : {len(anomalies)} "
      f"({len(anomalies)/len(df)*100:.1f}%)")

print("\nTop 10 anomalies :")

print(
    anomalies.nlargest(10, "lof_score")[
        [
            "drug",
            "side_effect",
            "period",
            "mention_count",
            "lof_score"
        ]
    ].to_string(index=False)
)

# ── JSON propre (sans category/source) ────────────────────────────
output = []

for _, row in anomalies.iterrows():

    output.append({
        "algorithm": "LOF",
        "drug": row["drug"],
        "side_effect": row["side_effect"],
        "period": row["period"],
        "mention_count": int(row["mention_count"]),
        "lof_score": round(float(row["lof_score"]), 4),
        "is_anomaly": True,
        "created_at": datetime.utcnow().isoformat()
    })

# ── Sauvegarde JSON ───────────────────────────────────────────────
output_file = "lof_anomalies.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\n✅ JSON créé : {output_file}")
print(f"Total anomalies sauvegardées : {len(output)}")