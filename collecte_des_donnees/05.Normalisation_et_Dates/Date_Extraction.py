import re, os, json, logging, hashlib
from datetime import datetime, date
from typing import Optional
from lxml import etree
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, ASCENDING

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("biosentry.axe3")

DATE_MIN = date(2019, 1, 1)
DATE_MAX = date(2026, 12, 31)
SUPPORTED_SOURCES = {"reddit", "openfda", "pubmed", "clinicaltrials", "drugs.com"}

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m",
    "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y",
]
_ISO_RE    = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEARMO_RE = re.compile(r"\b(\d{4})[/-](\d{2})\b")
_YEAR_RE   = re.compile(r"\b(20\d{2})\b")
_RANGE_RE  = re.compile(
    r"(\d{4}[/-]\d{2}[/-]\d{2})\s*(?:→|->|to|\s)\s*(\d{4}[/-]\d{2}[/-]\d{2})", re.I
)


def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI manquant dans .env")
    return MongoClient(uri)["biosentry_db"]


def ensure_indexes():
    ts = get_db()["temporal_signals"]
    ts.create_index([("source", ASCENDING), ("event_date", ASCENDING)], background=True)
    ts.create_index([("drug_name", ASCENDING), ("event_date", ASCENDING)], background=True)
    ts.create_index([("event_date", ASCENDING)], background=True)
    ts.create_index([("year_month", ASCENDING)], background=True)
    ts.create_index([("signal_type", ASCENDING), ("event_date", ASCENDING)], background=True)
    ts.create_index("_sig_hash", unique=True, sparse=True, background=True)


def _parse_date_str(raw: str) -> Optional[date]:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    for pattern, groups in [
        (_ISO_RE, lambda m: date(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        (_YEARMO_RE, lambda m: date(int(m.group(1)), int(m.group(2)), 1)),
        (_YEAR_RE, lambda m: date(int(m.group(1)), 1, 1)),
    ]:
        m = pattern.search(raw)
        if m:
            try:
                return groups(m)
            except ValueError:
                pass
    return None


def _parse_date_range(raw: str) -> tuple[Optional[date], Optional[date]]:
    if not raw:
        return None, None
    m = _RANGE_RE.search(raw)
    if m:
        return _parse_date_str(m.group(1)), _parse_date_str(m.group(2))
    return _parse_date_str(raw), None


def _in_window(d: Optional[date]) -> bool:
    return d is not None and DATE_MIN <= d <= DATE_MAX

def _year_month(d: date) -> str:
    return d.strftime("%Y-%m")

def _sig_hash(drug: str, source: str, event_date: str, extra: str = "") -> str:
    return hashlib.sha1(f"{drug}|{source}|{event_date}|{extra}".encode()).hexdigest()[:20]

def _make_signal(d: date, source: str, drug: str, signal_type: list, source_id: str, extra: dict) -> dict:
    return {
        "source": source, "drug_name": drug,
        "event_date": d.isoformat(),
        "event_date_obj": datetime(d.year, d.month, d.day),
        "year_month": _year_month(d), "year": d.year,
        "resolution": "month" if source == "pubmed" else "day",
        "signal_type": signal_type,
        "source_id": source_id,
        "date_start": d.isoformat(), "date_end": None, "is_range": False,
        "_sig_hash": _sig_hash(drug, source, d.isoformat(), source_id),
        **extra,
    }


def _extract_reddit(doc: dict) -> list[dict]:
    raw  = doc.get("review_date") or doc.get("date") or ""
    d    = _parse_date_str(str(raw))
    if not _in_window(d):
        return []
    drug = (doc.get("drug_name") or "").strip().lower()
    meta = doc.get("reddit_meta") or {}
    post_id = str(meta.get("post_id") or doc.get("_id") or "")
    sig = _make_signal(d, "reddit", drug, ["patient_review", "side_effect_report"], post_id,
                       {"subreddit": (meta.get("subreddit") or "").lower() or None, "_raw_date": str(raw)})
    return [sig]


def _extract_openfda(doc: dict) -> list[dict]:
    raw = doc.get("review_date") or doc.get("date") or ""
    d   = _parse_date_str(str(raw))
    if not _in_window(d):
        return []
    drug      = (doc.get("drug_name") or "").strip().lower()
    report_id = str(doc.get("report_id") or doc.get("_id") or "")
    return [_make_signal(d, "openfda", drug, ["side_effect_report"], report_id, {"_raw_date": str(raw)})]


def _extract_pubmed(doc: dict) -> list[dict]:
    drug      = (doc.get("drug_name") or "").strip().lower()
    pubmed_id = str(doc.get("pubmed_id") or doc.get("_id") or "")
    content   = doc.get("content") or ""
    xml_dates: list[date] = []

    if content:
        try:
            root = etree.fromstring(content.encode())
            for tag in ("PubDate", "ArticleDate", "DateCompleted", "DateRevised"):
                for node in root.iter(tag):
                    y, m, dy = node.findtext("Year"), node.findtext("Month") or "1", node.findtext("Day") or "1"
                    try:
                        m_int = int(m)
                    except ValueError:
                        try:
                            m_int = datetime.strptime(m[:3], "%b").month
                        except ValueError:
                            m_int = 1
                    if y:
                        try:
                            xml_dates.append(date(int(y), m_int, max(1, int(dy) if dy.isdigit() else 1)))
                        except ValueError:
                            pass
            for node in root.iter("MedlineDate"):
                d = _parse_date_str((node.text or "").strip())
                if d:
                    xml_dates.append(d)
        except Exception as e:
            logger.debug(f"[PUBMED] XML error ({pubmed_id}): {e}")

    range_raw = doc.get("date_range") or ""
    if range_raw:
        d_start, _ = _parse_date_range(str(range_raw))
        if d_start:
            xml_dates.append(d_start)

    seen, results = set(), []
    for d in xml_dates:
        if not _in_window(d) or d.isoformat() in seen:
            continue
        seen.add(d.isoformat())
        results.append(_make_signal(d, "pubmed", drug, ["literature_evidence"], pubmed_id,
                                    {"_raw_date": str(range_raw or content[:50])}))
    return results


def _extract_clinicaltrials(doc: dict) -> list[dict]:
    drug   = (doc.get("drug_name") or "").strip().lower()
    nct_id = str(doc.get("nct_id") or doc.get("_id") or "")
    results, seen = [], set()

    d_start      = _parse_date_str(str(doc.get("start_date") or ""))
    d_range_s, d_range_e = _parse_date_range(str(doc.get("date_range") or ""))
    primary      = d_start or d_range_s

    if primary and _in_window(primary):
        iso     = primary.isoformat()
        end_iso = d_range_e.isoformat() if d_range_e else None
        seen.add(iso)
        results.append({
            **_make_signal(primary, "clinicaltrials", drug, ["clinical_study"], nct_id,
                           {"nct_id": nct_id, "status": (doc.get("status") or "").lower() or None,
                            "date_end": end_iso, "is_range": end_iso is not None,
                            "_raw_date": f"{doc.get('start_date')} | {doc.get('date_range')}"}),
        })

    if d_range_e and _in_window(d_range_e) and d_range_e.isoformat() not in seen:
        iso_end = d_range_e.isoformat()
        seen.add(iso_end)
        results.append({
            **_make_signal(d_range_e, "clinicaltrials", drug, ["clinical_study", "study_end"], nct_id,
                           {"nct_id": nct_id, "status": (doc.get("status") or "").lower() or None,
                            "date_start": primary.isoformat() if primary else None,
                            "date_end": iso_end, "is_range": True,
                            "_sig_hash": _sig_hash(drug, "clinicaltrials", iso_end + "_end", nct_id),
                            "_raw_date": str(doc.get("date_range") or "")}),
        })
    return results


def _extract_drugscom(doc: dict) -> list[dict]:
    raw = doc.get("review_date") or doc.get("date") or ""
    d   = _parse_date_str(str(raw))
    if not _in_window(d):
        return []
    drug      = (doc.get("drug_name") or "").strip().lower()
    review_id = str(doc.get("review_key") or doc.get("_id") or "")
    return [_make_signal(d, "drugs.com", drug, ["patient_review", "side_effect_report"], review_id,
                         {"_raw_date": str(raw)})]


_EXTRACTORS = {
    "reddit": _extract_reddit, "openfda": _extract_openfda,
    "pubmed": _extract_pubmed, "clinicaltrials": _extract_clinicaltrials,
    "drugs.com": _extract_drugscom,
}


def extract_temporal_signals(doc: dict) -> list[dict]:
    source = (doc.get("source") or "").lower().strip()
    fn = _EXTRACTORS.get(source)
    if not fn:
        return []
    try:
        signals = fn(doc)
    except Exception as e:
        logger.error(f"[AXE3] Erreur [{source}]: {e}", exc_info=True)
        return []
    for s in signals:
        s["conditions"] = doc.get("conditions") or []
        s["url"]        = doc.get("url") or None
    return signals


def is_signal_already_stored(sig_hash: str) -> bool:
    return get_db()["temporal_signals"].count_documents({"_sig_hash": sig_hash}, limit=1) > 0


def load_stored_hashes(source_filter: list[str] | None = None) -> set[str]:
    col   = get_db()["temporal_signals"]
    query = {"source": {"$in": source_filter}} if source_filter else {}
    hashes = {d["_sig_hash"] for d in col.find(query, {"_sig_hash": 1}) if d.get("_sig_hash")}
    logger.info(f"[CHECKPOINT] {len(hashes)} signaux en base")
    return hashes


def save_signals(signals: list[dict]) -> None:
    if not signals:
        return
    col = get_db()["temporal_signals"]
    ops = [UpdateOne({"_sig_hash": s["_sig_hash"]}, {"$set": s}, upsert=True)
           for s in signals if s.get("_sig_hash")]
    if ops:
        res = col.bulk_write(ops, ordered=False)
        logger.info(f"[MONGO] insérés={res.upserted_count} | maj={res.modified_count}")


def load_raw_sources(source_filter: list[str] | None = None, limit: int = 0) -> list[dict]:
    db    = get_db()
    query = {"source": {"$in": source_filter}} if source_filter else {}
    cur   = db["raw_sources"].find(query)
    if limit:
        cur = cur.limit(limit)
    docs = list(cur)
    logger.info(f"[MONGO] {len(docs)} docs chargés")
    return docs


def run_temporal_pipeline(
    source_filter: list[str] | None = None,
    limit: int = 0,
    dry_run: bool = False,
    batch_size: int = 500,
    skip_done: bool = True,
    parallel_mode: bool = False,
) -> dict[str, int]:
    ensure_indexes()

    effective_sources = [s for s in (source_filter or list(SUPPORTED_SOURCES)) if s in SUPPORTED_SOURCES]
    if not effective_sources:
        logger.warning("[AXE3] Aucune source supportée.")
        return {"extracted": 0, "skipped_window": 0, "skipped_duplicate": 0, "saved": 0, "errors": 0}

    raw_docs = load_raw_sources(effective_sources, limit)
    stored_hashes: set[str] = set()
    if skip_done and not dry_run and not parallel_mode:
        stored_hashes = load_stored_hashes(effective_sources)

    counters = {"extracted": 0, "skipped_window": 0, "skipped_duplicate": 0, "saved": 0, "errors": 0}
    batch: list[dict] = []

    for doc in raw_docs:
        try:
            signals = extract_temporal_signals(doc)
        except Exception as e:
            logger.error(f"[AXE3] {e}", exc_info=True)
            counters["errors"] += 1
            continue

        if not signals:
            counters["skipped_window"] += 1
            continue

        for sig in signals:
            counters["extracted"] += 1
            h = sig.get("_sig_hash", "")
            if skip_done and not dry_run:
                if parallel_mode:
                    if is_signal_already_stored(h):
                        counters["skipped_duplicate"] += 1
                        continue
                else:
                    if h in stored_hashes:
                        counters["skipped_duplicate"] += 1
                        continue
                    stored_hashes.add(h)
            batch.append(sig)
            counters["saved"] += 1
            if not dry_run and len(batch) >= batch_size:
                save_signals(batch)
                batch.clear()

    if not dry_run and batch:
        save_signals(batch)

    logger.info(
        f"[AXE3] extraits={counters['extracted']} | hors-fenêtre={counters['skipped_window']} | "
        f"doublons={counters['skipped_duplicate']} | sauvegardés={counters['saved']} | erreurs={counters['errors']}"
    )
    return counters


if __name__ == "__main__":
    import argparse
    VALID_SOURCES = sorted(SUPPORTED_SOURCES)

    p = argparse.ArgumentParser(description="Bio-Sentry Axe 3 — Normalisation temporelle 2019–2026")
    p.add_argument("--source",     metavar="SOURCE", choices=VALID_SOURCES)
    p.add_argument("--sources",    nargs="*", metavar="SOURCE")
    p.add_argument("--limit",      type=int, default=0)
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--force",      action="store_true")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()

    if args.source:
        sources, parallel_mode = [args.source], True
    elif args.sources:
        sources, parallel_mode = args.sources, False
    else:
        sources, parallel_mode = VALID_SOURCES, False

    counters = run_temporal_pipeline(
        source_filter=sources, limit=args.limit,
        dry_run=args.dry_run, batch_size=args.batch_size,
        skip_done=not args.force, parallel_mode=parallel_mode,
    )
    print(
        f"\n{'─'*50}\n"
        f"  extraits    : {counters['extracted']}\n"
        f"  hors-fenêtre: {counters['skipped_window']}\n"
        f"  doublons    : {counters['skipped_duplicate']}\n"
        f"  sauvegardés : {counters['saved']}\n"
        f"  erreurs     : {counters['errors']}\n"
        f"{'─'*50}\n"
    )
