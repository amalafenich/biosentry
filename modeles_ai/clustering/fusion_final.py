import pandas as pd
import json

# =====================================================
# 1. CHARGEMENT DES 3 JSON
# =====================================================
print("=== CHARGEMENT DES 3 FICHIERS JSON ===")

with open("kmeans_results.json",  "r", encoding="utf-8") as f:
    kmeans_data = json.load(f)

with open("agglo_results.json",   "r", encoding="utf-8") as f:
    agglo_data = json.load(f)

with open("hdbscan_results.json", "r", encoding="utf-8") as f:
    hdbscan_data = json.load(f)

df_kmeans  = pd.DataFrame(kmeans_data)
df_agglo   = pd.DataFrame(agglo_data)
df_hdbscan = pd.DataFrame(hdbscan_data)

print(f"KMeans  : {len(df_kmeans)} lignes")
print(f"Agglo   : {len(df_agglo)} lignes")
print(f"HDBSCAN : {len(df_hdbscan)} lignes")

# =====================================================
# 2. FUSION SUR (drug, side_effect, period)
# =====================================================
print("\n=== FUSION ===")

# Base : KMeans
combined = df_kmeans[["drug", "side_effect", "period", "kmeans_cluster"]].copy()

# Join Agglomerative
combined = combined.merge(
    df_agglo[["drug", "side_effect", "period", "agglo_cluster"]],
    on=["drug", "side_effect", "period"],
    how="left"
)

# Join HDBSCAN
combined = combined.merge(
    df_hdbscan[["drug", "side_effect", "period", "hdbscan_cluster"]],
    on=["drug", "side_effect", "period"],
    how="left"
)

# Remplir les valeurs manquantes éventuelles
combined["agglo_cluster"]   = combined["agglo_cluster"].fillna(-2).astype(int)
combined["hdbscan_cluster"] = combined["hdbscan_cluster"].fillna(-2).astype(int)

print(f"Dataset fusionné : {len(combined)} lignes")
print(f"Colonnes         : {list(combined.columns)}")

# =====================================================
# 3. APERÇU
# =====================================================
print("\nAperçu (10 premières lignes) :")
print(combined.head(10).to_string(index=False))

# =====================================================
# 4. VÉRIFICATION — doublons
# =====================================================
dupes = combined.duplicated(
    subset=["drug", "side_effect", "period"]
).sum()

print(f"\nDoublons (drug, side_effect, period) : {dupes}")

# =====================================================
# 5. EXPORT JSON FINAL
# =====================================================
output = combined.to_dict(orient="records")

with open("combined_clusters.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\n✅ JSON créé : combined_clusters.json ({len(output)} lignes)")

# =====================================================
# 6. EXPORT CSV (optionnel)
# =====================================================
combined.to_csv("combined_clusters.csv", index=False, encoding="utf-8")

print("✅ CSV créé  : combined_clusters.csv")

print("\n✅ Fusion terminée.")
