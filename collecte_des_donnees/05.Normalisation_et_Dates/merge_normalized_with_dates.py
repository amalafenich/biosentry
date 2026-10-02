import os, json, logging
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, ASCENDING

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("biosentry.enrich")

_COLS_TO_DROP        = {"scraped_at", "_id"}
_TEMPORAL_FIELDS     = ("event_date", "year_month", "signal_type", "is_range")
_URL_UNRELIABLE      = {"openfda", "drugs.com", "reddit"}
_SOURCE_ID_PRIORITY  = {"reddit", "drugs.com", "openfda", "pubmed", "clinicaltrials"}
_EMPTY_TEMPORAL      = {"event_date": None, "year_month": None, "signal_type": [], "is_range": False}

ALL_SOURCES = [
    "medlineplus", "clinicaltrials", "drugs.com", "drugs_com",
    "openfda", "pubmed", "vigiaccess", "webmd", "reddit",
]


def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI manquant dans .env")
    return MongoClient(uri)["biosentry_db"]


def ensure_indexes():
    col = get_db()["enriched_signals"]
    col.create_index([("drug_name", ASCENDING), ("source", ASCENDING)], background=True)
    col.create_index([("event_date", ASCENDING)], background=True)
    col.create_index([("year_month", ASCENDING)], background=True)
    col.create_index([("signal_type", ASCENDING), ("event_date", ASCENDING)], background=True)
    col.create_index([("source", ASCENDING), ("event_date", ASCENDING)], background=True)


def load_normalised(source_filter: list[str] | None = None) -> list[dict]:
    db    = get_db()
    query = {"source": {"$in": source_filter}} if source_filter else {}
    docs  = []
    for d in db["normalised_sources"].find(query):
        for col in _COLS_TO_DROP:
            d.pop(col, None)
        docs.append(d)
    logger.info(f"[MONGO] {len(docs)} docs chargés depuis normalised_sources")
    return docs


def load_temporal_index(source_filter: list[str] | None = None) -> dict:
    db    = get_db()
    query = {"source": {"$in": source_filter}} if source_filter else {}
    url_idx, ds_idx = {}, {}

    for sig in db["temporal_signals"].find(query):
        sig.pop("_id", None)
        url       = (sig.get("url") or "").strip()
        drug      = (sig.get("drug_name") or "").strip().lower()
        source    = (sig.get("source") or "").strip().lower()
        source_id = str(sig.get("source_id") or "").strip()

        if url:
            url_idx.setdefault(url, []).append(sig)
        if drug and source:
            ds_idx.setdefault((drug, source), []).append(sig)
        if drug and source and source_id:
            ds_idx.setdefault((drug, source, source_id), []).append(sig)

    logger.info(f"[MONGO] temporal indexé — url_keys={len(url_idx)} | ds_keys={len(ds_idx)}")
    return {"url": url_idx, "drug_source": ds_idx}


def _pick_best_signal(signals: list[dict]) -> dict:
    pool = [s for s in signals if "study_end" not in (s.get("signal_type") or [])] or signals
    return sorted(pool, key=lambda s: s.get("event_date") or "0000-00-00", reverse=True)[0]


def join_temporal(norm_doc: dict, temporal_index: dict) -> dict:
    url       = (norm_doc.get("url") or "").strip()
    drug      = (norm_doc.get("drug_name") or "").strip().lower()
    source    = (norm_doc.get("source") or "").strip().lower()
    source_id = str(norm_doc.get("source_id") or "").strip()
    signals   = []

    if drug and source and source_id:
        signals = temporal_index["drug_source"].get((drug, source, source_id), [])
    if not signals and url and source not in _URL_UNRELIABLE:
        signals = temporal_index["url"].get(url, [])
    if not signals:
        signals = temporal_index["drug_source"].get((drug, source), [])

    if not signals:
        return {**norm_doc, **_EMPTY_TEMPORAL}

    best = _pick_best_signal(signals)
    patch = {f: best.get(f, _EMPTY_TEMPORAL[f]) for f in _TEMPORAL_FIELDS}
    patch["is_range"] = bool(patch.get("is_range", False))
    return {**norm_doc, **patch}


def save_enriched(docs: list[dict]) -> None:
    if not docs:
        return
    col = get_db()["enriched_signals"]
    ops = []
    for d in docs:
        drug, src, url, sid = d.get("drug_name", ""), d.get("source", ""), d.get("url"), d.get("source_id")
        filt = {"drug_name": drug, "source": src}
        if src in _SOURCE_ID_PRIORITY and sid:
            filt["source_id"] = sid
        elif url and src not in _URL_UNRELIABLE:
            filt["url"] = url
        elif sid:
            filt["source_id"] = sid
        else:
            filt["event_date"] = d.get("event_date")
        ops.append(UpdateOne(filt, {"$set": d}, upsert=True))
    res = col.bulk_write(ops, ordered=False)
    logger.info(f"[MONGO:enriched] insérés={res.upserted_count} | maj={res.modified_count}")


def run_enrich_pipeline(
    source_filter: list[str] | None = None,
    dry_run: bool = False,
    batch_size: int = 500,
    force: bool = False,
) -> dict[str, int]:
    ensure_indexes()

    if force and not dry_run:
        col = get_db()["enriched_signals"]
        n = col.count_documents({})
        col.drop()
        logger.info(f"[FORCE] enriched_signals vidée ({n} docs supprimés)")

    norm_docs      = load_normalised(source_filter)
    temporal_index = load_temporal_index(source_filter)
    counters = {"joined": 0, "no_temporal": 0, "saved": 0, "errors": 0}
    batch: list[dict] = []

    for norm_doc in norm_docs:
        try:
            enriched = join_temporal(norm_doc, temporal_index)
        except Exception as e:
            logger.error(f"[ENRICH] {e}", exc_info=True)
            counters["errors"] += 1
            continue

        counters["joined" if enriched.get("event_date") else "no_temporal"] += 1
        batch.append(enriched)
        counters["saved"] += 1

        if not dry_run and len(batch) >= batch_size:
            save_enriched(batch)
            batch.clear()

    if not dry_run and batch:
        save_enriched(batch)

    logger.info(
        f"[ENRICH] jointures={counters['joined']} | sans_temporel={counters['no_temporal']} | "
        f"sauvegardés={counters['saved']} | erreurs={counters['errors']}"
    )
    return counters


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Bio-Sentry Enrich — Fusion normalised × temporal")
    p.add_argument("--source",     metavar="SOURCE", choices=ALL_SOURCES)
    p.add_argument("--sources",    nargs="*", metavar="SOURCE")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--force",      action="store_true")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()

    sources  = [args.source] if args.source else (args.sources or ALL_SOURCES)
    counters = run_enrich_pipeline(
        source_filter=sources, dry_run=args.dry_run,
        batch_size=args.batch_size, force=args.force,
    )
    print(
        f"\n{'─'*50}\n"
        f"  jointures   : {counters['joined']}\n"
        f"  sans tempo  : {counters['no_temporal']}\n"
        f"  sauvegardés : {counters['saved']}\n"
        f"  erreurs     : {counters['errors']}\n"
        f"{'─'*50}\n"
    )
