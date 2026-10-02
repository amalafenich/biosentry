from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from pymongo import MongoClient
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))["biosentry_db"]

print("=== CHARGEMENT DATA_AGGREGATION ===")
df = pd.DataFrame(db["data_aggregation"].find({}, {"_id": 0}))
print(f"Documents chargés : {len(df)}")

# ── Features engineering ──────────────────────────────────────
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

features = ["mention_count", "mean_count", "std_count", "total_mentions", "num_periods", "month"]
X = df[features].fillna(0)

# ── Normalisation ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Isolation Forest ──────────────────────────────────────────
print("Calcul Isolation Forest en cours...")
iforest = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)
iforest.fit(X_scaled)
scores = -iforest.score_samples(X_scaled)

# ── Seuil AUTOMATIQUE IQR Tukey ───────────────────────────────
Q1 = pd.Series(scores).quantile(0.25)
Q3 = pd.Series(scores).quantile(0.75)
IQR = Q3 - Q1
seuil_auto = Q3 + 1.5 * IQR

print(f"\nSeuil IQR Tukey calculé : {seuil_auto:.4f}")
print(f"  Q1={Q1:.4f} | Q3={Q3:.4f} | IQR={IQR:.4f}")

df["if_score"]   = scores
df["if_anomaly"] = (pd.Series(scores) > seuil_auto).astype(int)

anomalies = df[df["if_anomaly"] == 1].copy()

print(f"\nTotal anomalies : {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}%)")
print(f"→ Seuil calculé automatiquement par IQR Tukey sur les scores IF")

print(f"\nTop 10 anomalies :")
print(anomalies.nlargest(10, "if_score")[
    ["drug", "side_effect", "period", "mention_count", "if_score"]
].to_string(index=False))

# ── JSON seulement ────────────────────────────────────────────
df[["drug", "side_effect", "period", "mention_count",
    "if_score", "if_anomaly"]].to_json(
    "test_iforest.json", orient="records", indent=2, force_ascii=False
)

print(f"\n✅ JSON créé : test_iforest.json")
print(f"   Total     : {len(df)}")
print(f"   Anomalies : {int(df['if_anomaly'].sum())}")