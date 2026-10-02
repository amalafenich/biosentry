import requests
import time
import random
import os
import argparse
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()

LIMIT_PER_DRUG = 25
POSTS_PER_PAGE = 25
REQUEST_DELAY  = 2.5
MAX_PAGES      = 4

SUBREDDITS = [
    "AskDocs", "pharmacy", "Medications", "druginteractions",
    "ChronicPain", "medical", "askpharmacist", "Nootropics", "PrescriptionDrugs",
]

SIDE_EFFECT_TERMS_A = [
    "nausea", "vomiting", "headache", "dizziness", "fatigue", "rash",
    "diarrhea", "constipation", "stomach", "abdominal", "fever",
    "insomnia", "anxiety", "blurred vision", "dry mouth", "sweating",
    "palpitations", "shortness of breath", "itching", "swelling",
    "hair loss", "muscle pain", "muscle ache", "joint pain", "chest pain",
    "high blood pressure", "hypertension", "bleeding", "allergic",
    "drowsiness", "sleepy", "sleepiness", "weight gain", "weight loss",
    "cramps", "bloating", "heartburn", "acid reflux", "back pain",
    "kidney", "liver", "tingling", "numbness", "confusion",
    "depression", "mood", "irritable", "panic", "tremor", "shakiness",
    "bruising", "dark urine", "yellow skin", "jaundice",
]

SIDE_EFFECT_TERMS_B = [
    "side effect", "side effects", "adverse", "reaction", "caused by",
    "due to", "from taking", "after taking", "since taking", "since starting",
    "because of", "stopped taking", "stopped the", "quit taking",
    "experiencing", "experienced", "suffering from", "made me",
    "gave me", "causing", "causes", "triggered", "developed",
    "started having", "started getting", "noticed", "symptom", "symptoms",
    "prescribed", "taking it", "on it", "on the medication",
    "withdrawal", "tolerance", "dependence",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI manquant dans .env")
    return MongoClient(uri)["biosentry_db"]


def load_drug_list():
    docs = list(get_db()["drug_list"].find({}, {"_id": 0, "name": 1, "category": 1}))
    seen, result = set(), []
    for d in docs:
        name = d.get("name", "").strip().lower()
        if name and name not in seen:
            seen.add(name)
            result.append({"drug_name": name, "category": d.get("category", "")})
    print(f"[DRUG_LIST] {len(result)} médicaments")
    return result


def get_existing_count(slug):
    return get_db()["raw_sources"].count_documents({"source": "reddit", "drug_name": slug})


def save_to_mongo(records):
    if not records:
        return
    ops = [
        UpdateOne(
            {"drug_name": r["drug_name"], "review_key": r["review_key"]},
            {"$setOnInsert": r},
            upsert=True,
        )
        for r in records
    ]
    res = get_db()["raw_sources"].bulk_write(ops, ordered=False)
    print(f"  [MONGO] Insérés: {res.upserted_count} | Existants: {res.matched_count}")


def ensure_indexes():
    col = get_db()["raw_sources"]
    try:
        col.drop_index("reddit_drug_review_unique")
    except Exception:
        pass
    col.create_index(
        [("drug_name", 1), ("review_key", 1)],
        unique=True,
        partialFilterExpression={"source": "reddit"},
        name="reddit_drug_review_unique",
    )
    for field in ["drug_name", "category", "scraped_at", "confidence"]:
        col.create_index(field)


def is_relevant(text, drug_slug):
    tl = text.lower()
    return (
        drug_slug in tl
        and any(t in tl for t in SIDE_EFFECT_TERMS_A)
        and any(t in tl for t in SIDE_EFFECT_TERMS_B)
    )


def assess_confidence(score, subreddit):
    trusted = {"askdocs", "pharmacy", "askpharmacist", "druginteractions"}
    sub = subreddit.lower()
    if score >= 50 and sub in trusted:
        return "high"
    if score >= 10 or sub in trusted:
        return "medium"
    return "low"


def safe_get(url, params, retries=4):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 60)))
                continue
            if resp.status_code in (403, 404):
                return None
            if resp.status_code != 200:
                time.sleep(10 * attempt)
                continue
            return resp.json()
        except requests.RequestException as e:
            print(f"  [RÉSEAU] {e}")
            time.sleep(5 * attempt)
    return None


def search_subreddit(subreddit, drug_slug, after=None):
    data = safe_get(
        f"https://www.reddit.com/r/{subreddit}/search.json",
        {"q": f"{drug_slug} side effects", "sort": "relevance", "t": "all",
         "limit": POSTS_PER_PAGE, "restrict_sr": "true", **({"after": after} if after else {})},
    )
    if not data:
        return [], None

    listing  = data.get("data", {})
    records  = []

    for wrapper in listing.get("children", []):
        post      = wrapper.get("data", {})
        full_text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
        if not is_relevant(full_text, drug_slug):
            continue
        score   = post.get("score", 0)
        created = post.get("created_utc", 0)
        records.append({
            "source"     : "reddit",
            "category"   : "",
            "drug_name"  : drug_slug,
            "condition"  : "",
            "rating"     : None,
            "confidence" : assess_confidence(score, subreddit),
            "review_date": datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d") if created else "",
            "scraped_at" : datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "url"        : f"https://reddit.com{post.get('permalink', '')}",
            "review_text": full_text[:1000],
            "review_key" : drug_slug + full_text[:80],
            "reddit_meta": {"post_id": post.get("id", ""), "score": score, "subreddit": subreddit, "type": "post"},
        })

    return records, listing.get("after")


def fetch_comments(subreddit, post_id, drug_slug, category):
    data = safe_get(f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json", {})
    if not data or not isinstance(data, list) or len(data) < 2:
        return []

    records = []
    for wrapper in data[1].get("data", {}).get("children", []):
        c    = wrapper.get("data", {})
        body = c.get("body", "")
        if len(body) < 40 or not is_relevant(body, drug_slug):
            continue
        score   = c.get("score", 0)
        created = c.get("created_utc", 0)
        records.append({
            "source"     : "reddit",
            "category"   : category,
            "drug_name"  : drug_slug,
            "condition"  : "",
            "rating"     : None,
            "confidence" : assess_confidence(score, subreddit),
            "review_date": datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d") if created else "",
            "scraped_at" : datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "url"        : f"https://reddit.com/r/{subreddit}/comments/{post_id}/",
            "review_text": body[:1000],
            "review_key" : drug_slug + body[:80],
            "reddit_meta": {"comment_id": c.get("id", ""), "post_id": post_id, "score": score, "subreddit": subreddit, "type": "comment"},
        })
    return records


def fetch_drug_reviews(drug_slug, category, already_saved=0):
    collected = already_saved

    for subreddit in SUBREDDITS:
        if collected >= LIMIT_PER_DRUG:
            break
        print(f"    r/{subreddit} …", end=" ", flush=True)
        after, sub_count = None, 0

        for _ in range(MAX_PAGES):
            if collected >= LIMIT_PER_DRUG:
                break
            records, after = search_subreddit(subreddit, drug_slug, after)
            for post in records[:3]:
                if collected >= LIMIT_PER_DRUG:
                    break
                records.extend(fetch_comments(subreddit, post["reddit_meta"]["post_id"], drug_slug, category))
                time.sleep(REQUEST_DELAY)
            for r in records:
                if not r["category"]:
                    r["category"] = category
            save_to_mongo(records)
            collected += len(records)
            sub_count += len(records)
            time.sleep(REQUEST_DELAY)
            if not after or not records:
                break

        print(f"{sub_count} avis")

    return collected - already_saved


def run(reset=False):
    db = get_db()
    if reset:
        db["raw_sources"].delete_many({"source": "reddit"})

    drug_docs    = load_drug_list()
    ensure_indexes()

    fully_done, remaining = [], []
    for d in drug_docs:
        count = get_existing_count(d["drug_name"])
        if count >= LIMIT_PER_DRUG:
            fully_done.append(d["drug_name"])
        else:
            remaining.append({**d, "_existing_count": count})

    print(f"  {len(fully_done)}/{len(drug_docs)} complétés | {len(remaining)} restants")

    for i, drug in enumerate(remaining, 1):
        print(f"[{i:>3}/{len(remaining)}] {drug['drug_name']} ({drug['category']})")
        new_count = fetch_drug_reviews(drug["drug_name"], drug["category"], drug["_existing_count"])
        print(f"  -> {new_count} nouveaux avis\n")
        time.sleep(random.uniform(5.0, 12.0))

    total = db["raw_sources"].count_documents({"source": "reddit"})
    print(f"\n[DONE] {total} avis Reddit au total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run(reset=args.reset)
