import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os
import time

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))["biosentry_db"]

CT_BASE = "https://clinicaltrials.gov/api/v2/studies"

DATE_START = "2019-01-01"
DATE_END   = "2026-12-31"


def get_drug_names():
    drugs = list(db["drug_list"].find({}, {"name": 1, "category": 1}))
    print(f"📋 {len(drugs)} médicaments chargés depuis drug_list")
    return drugs


def search_trials(drug_name, max_results=10):
    params = {
        "query.intr":  drug_name,
        "query.term":  "adverse effects",
        "filter.advanced": f"AREA[StartDate]RANGE[{DATE_START},{DATE_END}]",
        "pageSize":    max_results,
        "format":      "json",
        "fields":      "NCTId,BriefTitle,BriefSummary,Condition,InterventionName,StartDate,OverallStatus",
    }
    try:
        r = requests.get(CT_BASE, params=params, timeout=15)
        if r.status_code != 200:
            return []
        studies = r.json().get("studies", [])
        return studies
    except Exception as e:
        print(f"  ⚠️  Erreur search: {e}")
        return []


def already_exists(nct_id):
    return db["raw_sources"].find_one(
        {"source": "clinicaltrials", "nct_id": nct_id}
    ) is not None


def save_trial(study, drug_name, category):
    proto = study.get("protocolSection", {})
    id_module      = proto.get("identificationModule", {})
    desc_module    = proto.get("descriptionModule", {})
    status_module  = proto.get("statusModule", {})
    cond_module    = proto.get("conditionsModule", {})
    interv_module  = proto.get("armsInterventionsModule", {})

    nct_id = id_module.get("nctId", "")
    if not nct_id:
        return

    doc = {
        "source":       "clinicaltrials",
        "nct_id":       nct_id,
        "drug_name":    drug_name,
        "category":     category,
        "title":        id_module.get("briefTitle", ""),
        "summary":      desc_module.get("briefSummary", ""),
        "conditions":   cond_module.get("conditions", []),
        "status":       status_module.get("overallStatus", ""),
        "start_date":   status_module.get("startDateStruct", {}).get("date", ""),
        "url":          f"https://clinicaltrials.gov/study/{nct_id}",
        "date_range":   f"{DATE_START} → {DATE_END}",
        "scraped_at":   datetime.utcnow(),
    }

    db["raw_sources"].update_one(
        {"source": "clinicaltrials", "nct_id": nct_id},
        {"$set": doc},
        upsert=True
    )


def run():
    drugs = get_drug_names()
    total = 0

    for i, drug in enumerate(drugs):
        name     = drug["name"]
        category = drug["category"]

        print(f"\n[{i+1}/{len(drugs)}] 🔍 {name} ({category})")
        studies = search_trials(name, max_results=10)

        if not studies:
            print(f"  ⚠️  Aucun essai clinique trouvé")
            continue

        print(f"  📦 {len(studies)} essais trouvés (2019-2026)")

        for study in studies:
            proto  = study.get("protocolSection", {})
            nct_id = proto.get("identificationModule", {}).get("nctId", "")

            if already_exists(nct_id):
                print(f"  ⏩ {nct_id} déjà en base")
                continue

            save_trial(study, name, category)
            total += 1
            print(f"  ✅ {nct_id} sauvegardé")

        time.sleep(0.3)

    print(f"\n🎉 Terminé !")
    print(f"   Nouveaux essais : {total}")
    print(f"   Total raw_sources : {db['raw_sources'].count_documents({})}")


if __name__ == "__main__":
    run()