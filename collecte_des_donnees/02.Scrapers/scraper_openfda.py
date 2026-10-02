import requests
import time
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY        = os.getenv("OPENFDA_API_KEY", "*************************************")
BASE_URL       = "https://api.fda.gov/drug/event.json"
LIMIT_PER_DRUG = 150
SOURCE_NAME    = "openfda"


def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI manquant dans .env")
    return MongoClient(uri)["biosentry_db"]


def load_drug_list():
    db   = get_db()
    docs = list(db["drug_list"].find({}, {"_id": 0, "name": 1, "category": 1}))
    seen, result = set(), []
    for d in docs:
        name = d.get("name", "").strip().lower()
        if name and name not in seen:
            seen.add(name)
            result.append({"drug_name": name, "category": d.get("category", "")})
    print(f"[DRUG_LIST] {len(result)} médicaments")
    return result


def get_already_done():
    return set(get_db()["raw_sources"].distinct("drug_name", {"source": SOURCE_NAME}))


def ensure_indexes():
    col = get_db()["raw_sources"]
    try:
        col.drop_index("openfda_report_unique")
    except Exception:
        pass
    col.create_index(
        [("drug_name", 1), ("report_id", 1)],
        unique=True,
        partialFilterExpression={"source": SOURCE_NAME},
        name="openfda_report_unique",
    )
    for field in ["drug_name", "source", "category", "scraped_at", "confidence"]:
        col.create_index(field)


def save_to_mongo(records):
    if not records:
        return
    col = get_db()["raw_sources"]
    ops = [
        UpdateOne(
            {"source": SOURCE_NAME, "drug_name": r["drug_name"], "report_id": r["report_id"]},
            {"$setOnInsert": r},
            upsert=True,
        )
        for r in records
    ]
    res = col.bulk_write(ops, ordered=False)
    print(f"  [MONGO] Insérés: {res.upserted_count} | Existants: {res.matched_count}")


def build_params(drug, skip=0):
    return {
        "search" : f'patient.drug.medicinalproduct:"{drug}" AND receivedate:[20190101 TO 20261231]',
        "limit"  : min(LIMIT_PER_DRUG, 100),
        "skip"   : skip,
        "api_key": API_KEY,
    }


def parse_date(date_str):
    s = str(date_str).strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 and s.isdigit() else s


def extract_reactions(reactions):
    terms = [r.get("reactionmeddrapt", "").lower() for r in reactions if r.get("reactionmeddrapt")]
    return ", ".join(terms[:6]) if terms else "not specified"


def extract_drug_info(drugs, target):
    for d in drugs:
        if target.lower() in (d.get("medicinalproduct") or "").lower():
            return {"name": d.get("medicinalproduct", target), "dosage": d.get("drugdosagetext", ""), "indication": d.get("drugindication", "")}
    return {"name": target, "dosage": "", "indication": ""}


def fetch_fda_reports(drug_name, category):
    records, skip, fetched = [], 0, 0

    while fetched < LIMIT_PER_DRUG:
        try:
            resp = requests.get(BASE_URL, params=build_params(drug_name, skip), timeout=15)
            if resp.status_code == 429:
                time.sleep(60)
                continue
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [ERROR] {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for report in results:
            patient     = report.get("patient", {})
            reactions   = patient.get("reaction", [])
            symptom_str = extract_reactions(reactions)
            if symptom_str == "not specified":
                continue

            drug_info = extract_drug_info(patient.get("drug", []), drug_name)
            serious   = str(report.get("serious", "0"))

            records.append({
                "source"       : SOURCE_NAME,
                "category"     : category,
                "drug_name"    : drug_name,
                "scraped_at"   : datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "url"          : f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}",
                "confidence"   : "high" if serious == "1" else "medium",
                "report_id"    : report.get("safetyreportid", ""),
                "symptom"      : symptom_str,
                "review_date"  : parse_date(report.get("receivedate", "")),
                "indication"   : drug_info["indication"],
                "dosage"       : drug_info["dosage"],
                "serious"      : serious,
                "num_reactions": len(reactions),
            })
            fetched += 1

        total = data.get("meta", {}).get("results", {}).get("total", 0)
        skip += len(results)
        if skip >= total or fetched >= LIMIT_PER_DRUG:
            break
        time.sleep(0.3)

    return records


def print_stats():
    col = get_db()["raw_sources"]
    print(f"\n[STATS] Total: {col.count_documents({'source': SOURCE_NAME})} | "
          f"Sérieux: {col.count_documents({'source': SOURCE_NAME, 'serious': '1'})} | "
          f"Médicaments: {len(col.distinct('drug_name', {'source': SOURCE_NAME}))}")


def run(reset=False):
    print(f"[OpenFDA] LIMIT_PER_DRUG={LIMIT_PER_DRUG}")
    db = get_db()

    if reset:
        res = db["raw_sources"].delete_many({"source": SOURCE_NAME})
        print(f"  Supprimés: {res.deleted_count}")

    ensure_indexes()

    target_drugs = load_drug_list()
    if not target_drugs:
        return

    already_done = get_already_done()
    remaining    = [d for d in target_drugs if d["drug_name"] not in already_done]
    print(f"  {len(already_done)}/{len(target_drugs)} complétés | {len(remaining)} restants")

    for i, drug in enumerate(remaining, start=len(already_done) + 1):
        print(f"[{i:>3}/{len(target_drugs)}] {drug['drug_name']} ({drug['category']})")
        records = fetch_fda_reports(drug["drug_name"], drug["category"])
        save_to_mongo(records)
        print(f"  -> {len(records)} rapports")
        time.sleep(0.4)

    print_stats()


if __name__ == "__main__":
    run(reset=False)
