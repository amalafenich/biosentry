import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os
import time

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["biosentry_db"]
col = db["drug_list"]

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

CATEGORIES = {
    "NSAID":            "M01A",
    "Antibiotique":     "J01",
    "Antidepresseur":   "N06A",
    "Antihypertenseur": "C02",
    "Anticoagulant":    "B01A",
    "Statine":          "C10AA",
    "Antidiabetique":   "A10",
    "Corticoide":       "H02",
}


def get_drugs_by_atc(atc_code):
    r = requests.get(
        f"{RXNORM_BASE}/rxclass/classMembers.json",
        params={"classId": atc_code, "relaSource": "ATC"},
        timeout=10
    )
    if r.status_code != 200:
        return []
    members = r.json().get("drugMemberGroup", {}).get("drugMember", [])
    result = []
    for m in members:
        concept = m.get("minConcept", {})  # ← structure correcte
        rxcui = concept.get("rxcui")
        name  = concept.get("name", "").lower().strip()
        if rxcui and name:
            result.append({"rxcui": rxcui, "name": name})
    return result


def get_synonyms(rxcui):
    try:
        r = requests.get(f"{RXNORM_BASE}/rxcui/{rxcui}/allrelated.json", timeout=10)
        if r.status_code != 200:
            return []
        groups = r.json().get("allRelatedGroup", {}).get("conceptGroup", [])
        names = []
        for g in groups:
            for c in g.get("conceptProperties", []):
                name = c.get("name", "").lower().strip()
                if name:
                    names.append(name)
        return list(set(names))
    except Exception as e:
        print(f"  ⚠️  Erreur synonymes({rxcui}): {e}")
        return []


def build_drug_list():
    col.create_index("rxcui", unique=True)
    col.create_index("name")
    col.create_index("synonyms")
    col.create_index("category")

    total_inserted = 0
    total_updated  = 0

    for label, atc_code in CATEGORIES.items():
        print(f"\n🔍 {label} (ATC: {atc_code})")
        drugs = get_drugs_by_atc(atc_code)

        if not drugs:
            print(f"  ⚠️  Aucun médicament trouvé")
            continue

        print(f"  📦 {len(drugs)} médicaments trouvés")

        for drug in drugs:
            rxcui = drug["rxcui"]
            name  = drug["name"]

            synonyms = get_synonyms(rxcui)
            time.sleep(0.05)

            doc = {
                "rxcui":      rxcui,
                "name":       name,
                "category":   label,
                "atc_code":   atc_code,
                "synonyms":   synonyms,
                "updated_at": datetime.utcnow(),
            }

            result = col.update_one(
                {"rxcui": rxcui},
                {"$set": doc},
                upsert=True
            )

            if result.upserted_id:
                total_inserted += 1
            else:
                total_updated += 1

            print(f"  ✅ {name} ({rxcui}) — {len(synonyms)} synonymes")

    print(f"\n🎉 Terminé !")
    print(f"   Insérés    : {total_inserted}")
    print(f"   Mis à jour : {total_updated}")
    print(f"   Total drug_list : {col.count_documents({})}")


if __name__ == "__main__":
    build_drug_list()