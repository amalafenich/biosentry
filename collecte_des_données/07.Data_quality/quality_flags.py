from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np
import os

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))["biosentry_db"]


def run():
    print("📋 Chargement des données...")
    docs = list(db["normalised_sources"].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    print(f"   {len(df)} documents chargés")

    results = []

    for _, row in df.iterrows():
        flags = []
        score = 100  # score qualité de départ

        # ── 1. Complétude ──────────────────────────────────────
        missing_fields = []
        for field in ["drug_name", "source", "text_clean", "side_effects"]:
            val = row.get(field)
            if val is None or val == "" or val == []:
                missing_fields.append(field)
                score -= 10

        if missing_fields:
            flags.append(f"missing_fields:{','.join(missing_fields)}")

        # ── 2. Texte trop court ────────────────────────────────
        text = row.get("text_clean", "")
        text_len = len(str(text)) if text else 0
        if text_len < 50:
            flags.append("text_too_short")
            score -= 15
        elif text_len < 100:
            flags.append("text_short")
            score -= 5

        # ── 3. Pas d'effets secondaires détectés ──────────────
        side_effects = row.get("side_effects", [])
        if not side_effects or len(side_effects) == 0:
            flags.append("no_side_effects")
            score -= 10

        # ── 4. Source authority faible ────────────────────────
        authority = row.get("source_authority", 0)
        if authority == 0:
            flags.append("no_authority_score")
            score -= 5
        elif authority < 0.1:
            flags.append("low_authority")
            score -= 5

        # ── 5. Drug name manquant ──────────────────────────────
        drug_name = row.get("drug_name", "")
        if not drug_name:
            flags.append("missing_drug_name")
            score -= 20

        # ── 6. Source inconnue ─────────────────────────────────
        source = row.get("source", "")
        valid_sources = ["pubmed", "openfda", "clinicaltrials",
                        "drugs.com", "medlineplus", "reddit", "webmd"]
        if source not in valid_sources:
            flags.append("unknown_source")
            score -= 10

        # ── Qualité finale ─────────────────────────────────────
        score = max(0, score)
        if score >= 80:
            quality = "good"
        elif score >= 50:
            quality = "medium"
        else:
            quality = "poor"

        results.append({
            "drug_name":      drug_name,
            "source":         source,
            "url":            row.get("url", ""),
            "quality_score":  score,
            "quality_label":  quality,
            "flags":          flags,
            "text_length":    text_len,
            "has_side_effects": len(side_effects) > 0,
            "source_authority": authority,
            "checked_at":     datetime.utcnow(),
        })

    print(f"\n💾 Sauvegarde dans quality_flags...")
    col = db["quality_flags"]
    col.drop()
    col.insert_many(results)

    total  = len(results)
    good   = sum(1 for r in results if r["quality_label"] == "good")
    medium = sum(1 for r in results if r["quality_label"] == "medium")
    poor   = sum(1 for r in results if r["quality_label"] == "poor")

    print(f"\n🎉 Terminé !")
    print(f"   Total documents  : {total}")
    print(f"   ✅ Good           : {good}  ({good/total*100:.1f}%)")
    print(f"   ⚠️  Medium         : {medium} ({medium/total*100:.1f}%)")
    print(f"   ❌ Poor           : {poor}  ({poor/total*100:.1f}%)")

    print(f"\n📊 FLAGS LES PLUS FRÉQUENTS :")
    all_flags = []
    for r in results:
        all_flags.extend(r["flags"])
    flag_series = pd.Series(all_flags)
    print(flag_series.value_counts().head(10).to_string())


if __name__ == "__main__":
    run()