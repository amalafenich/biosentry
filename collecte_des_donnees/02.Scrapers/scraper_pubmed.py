import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os
import time

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))["biosentry_db"]

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL   = os.getenv("NCBI_EMAIL")

DATE_START = "2019/01/01"
DATE_END   = "2026/12/31"


def get_drug_names():
    drugs = list(db["drug_list"].find({}, {"name": 1, "category": 1}))
    print(f"📋 {len(drugs)} médicaments chargés depuis drug_list")
    return drugs


def search_pubmed(drug_name, max_results=20):
    params = {
        "db":       "pubmed",
        "term":     f"{drug_name}[tiab] AND adverse effects[tiab]",
        "retmax":   max_results,
        "retmode":  "json",
        "mindate":  DATE_START,
        "maxdate":  DATE_END,
        "datetype": "pdat",
        "api_key":  NCBI_API_KEY,
        "email":    NCBI_EMAIL,
    }
    try:
        r = requests.get(PUBMED_SEARCH, params=params, timeout=10)
        if r.status_code != 200:
            return []
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  ⚠️  Erreur search: {e}")
        return []


def fetch_article(pubmed_id):
    params = {
        "db":      "pubmed",
        "id":      pubmed_id,
        "retmode": "xml",
        "rettype": "abstract",
        "api_key": NCBI_API_KEY,
        "email":   NCBI_EMAIL,
    }
    try:
        r = requests.get(PUBMED_FETCH, params=params, timeout=10)
        if r.status_code != 200:
            return None
        return r.text
    except Exception as e:
        print(f"  ⚠️  Erreur fetch: {e}")
        return None


def already_exists(pubmed_id):
    return db["raw_sources"].find_one(
        {"source": "pubmed", "pubmed_id": pubmed_id}
    ) is not None


def save_article(pubmed_id, content, drug_name, category):
    doc = {
        "source":     "pubmed",
        "pubmed_id":  pubmed_id,
        "drug_name":  drug_name,
        "category":   category,
        "content":    content,
        "url":        f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
        "date_range": f"{DATE_START} → {DATE_END}",
        "scraped_at": datetime.utcnow(),
    }
    db["raw_sources"].update_one(
        {"source": "pubmed", "pubmed_id": pubmed_id},
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
        ids = search_pubmed(name, max_results=20)

        if not ids:
            print(f"  ⚠️  Aucun article trouvé")
            continue

        print(f"  📦 {len(ids)} articles trouvés (2019-2026)")

        for pubmed_id in ids:
            if already_exists(pubmed_id):
                print(f"  ⏩ {pubmed_id} déjà en base")
                continue
            content = fetch_article(pubmed_id)
            if content:
                save_article(pubmed_id, content, name, category)
                total += 1
                print(f"  ✅ {pubmed_id} sauvegardé")
            time.sleep(0.11)  # avec API key → 10 req/sec max

    print(f"\n🎉 Terminé !")
    print(f"   Nouveaux articles : {total}")
    print(f"   Total raw_sources : {db['raw_sources'].count_documents({})}")


if __name__ == "__main__":
    run()