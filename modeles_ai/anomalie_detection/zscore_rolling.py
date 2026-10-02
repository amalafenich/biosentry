from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from pymongo import MongoClient

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))["biosentry_db"]

print("=== CHARGEMENT DATA_AGGREGATION ===")
df = pd.DataFrame(db["data_aggregation"].find({}, {"_id": 0}))
print(f"Documents chargés : {len(df)}")

# ── Tri chronologique ─────────────────────────────────────────
df["period"] = df["period"].astype(str)
df = df.sort_values(["drug", "side_effect", "period"]).reset_index(drop=True)

results = []
window_stats = []  # pour voir les window calculés

for (drug, effect), group in df.groupby(["drug", "side_effect"]):
    group = group.copy().reset_index(drop=True)
    n = len(group)

    group["rolling_mean"]    = np.nan
    group["rolling_std"]     = np.nan
    group["z_score_rolling"] = 0.0
    group["is_anomaly"]      = False
    group["window_used"]     = 0
    group["method"]          = "trop_court"

    if n < 3:
        results.append(group)
        continue

    # ── Pas automatique basé sur les données ──────────────────
    if n < 4:
        window = 1
    elif n < 12:
        window = n // 3
    elif n < 24:
        window = n // 4
    else:
        window = n // 6

    group["window_used"] = window
    group["method"]      = f"zscore_rolling_auto_{window}mois"

    # ── calcul log window stats ────────────────────────────────
    window_stats.append({
        "drug"       : drug,
        "side_effect": effect,
        "n_periodes" : n,
        "window_auto": window
    })

    for i in range(window, n):
        historique      = group["mention_count"].iloc[i - window:i]
        valeur_actuelle = group["mention_count"].iloc[i]

        mean = historique.mean()
        std  = historique.std()

        if std == 0 or pd.isna(std):
            z = 0.0
        else:
            z = (valeur_actuelle - mean) / std

        group.at[i, "rolling_mean"]    = round(mean, 4)
        group.at[i, "rolling_std"]     = round(std, 4)
        group.at[i, "z_score_rolling"] = round(z, 4)

    # ── Seuil automatique IQR Tukey ───────────────────────────
    z_vals = group["z_score_rolling"].iloc[window:]

    if len(z_vals) > 0:
        Q1    = z_vals.quantile(0.25)
        Q3    = z_vals.quantile(0.75)
        IQR   = Q3 - Q1
        seuil = Q3 + 1.5 * IQR

        group["is_anomaly"] = group["z_score_rolling"] > seuil

    results.append(group)

df_result = pd.concat(results).reset_index(drop=True)
anomalies = df_result[df_result["is_anomaly"] == True]

print(f"\nTotal anomalies temporelles : {len(anomalies)} ({len(anomalies)/len(df_result)*100:.1f}%)")
print(f"→ Pas calculé automatiquement selon la taille de chaque groupe")

# ── Stats sur les windows utilisés ───────────────────────────
df_windows = pd.DataFrame(window_stats)
print(f"\n=== DISTRIBUTION DES WINDOW AUTOMATIQUES ===")
print(df_windows["window_auto"].value_counts().sort_index().to_string())

print(f"\nTop 10 anomalies temporelles :")
print(anomalies.nlargest(10, "z_score_rolling")[
    ["drug", "side_effect", "period", "mention_count", 
     "rolling_mean", "z_score_rolling", "window_used"]
].to_string(index=False))

# ── Sauvegarde JSON ───────────────────────────────────────────
df_result[[
    "drug", "side_effect", "period",
    "mention_count", "rolling_mean", "rolling_std",
    "z_score_rolling", "is_anomaly", "window_used", "method"
]].to_json(
    "test_zscore_rolling.json",
    orient="records", indent=2, force_ascii=False
)

print(f"\n✅ JSON créé : test_zscore_rolling.json")
print(f"   Total     : {len(df_result)}")
print(f"   Anomalies : {len(anomalies)}")