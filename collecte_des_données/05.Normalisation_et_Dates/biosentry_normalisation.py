import re
import html
import json
import os
import hashlib
import logging
from lxml import etree
from dotenv import load_dotenv
import spacy
from pymongo import MongoClient, UpdateOne
from nlp_biobert import analyse_text, NLPResult, normalise_term
from hits_authority import run_hits_pipeline, get_source_score, HITSScore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("biosentry.v5.4")

def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI manquant dans .env")
    return MongoClient(uri)["biosentry_db"]

_DRUG_CLASS_CACHE: dict[str, str] | None = None

def _get_drug_class_cache() -> dict[str, str]:
    global _DRUG_CLASS_CACHE
    if _DRUG_CLASS_CACHE is not None:
        return _DRUG_CLASS_CACHE
    _DRUG_CLASS_CACHE = {}
    try:
        for doc in get_db()["drug_list"].find({}, {"name": 1, "synonyms": 1, "category": 1}):
            cat  = (doc.get("category") or "").strip() or None
            name = (doc.get("name") or "").lower().strip()
            if name and cat:
                _DRUG_CLASS_CACHE[name] = cat
            for syn in (doc.get("synonyms") or []):
                s = syn.lower().strip()
                if s and cat:
                    _DRUG_CLASS_CACHE[s] = cat
    except Exception as e:
        logger.warning(f"[DRUG_LIST] {e}")
    return _DRUG_CLASS_CACHE

def _get_drug_class(drug_name: str) -> str | None:
    return _get_drug_class_cache().get(drug_name.lower().strip())

def _content_hash(r: dict) -> str:
    return hashlib.sha1(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:16]

def save_normalised(records: list[dict]) -> None:
    if not records:
        return
    col = get_db()["normalised_sources"]
    ops = []
    SOURCE_ID_PRIORITY = {"reddit", "drugs.com", "openfda", "pubmed", "clinicaltrials"}
    for r in records:
        if not r.get("side_effects") and not r.get("symptoms"):
            continue
        source = r.get("source", "")
        filt   = {"drug_name": r["drug_name"], "source": source}
        if source in SOURCE_ID_PRIORITY and r.get("source_id"):
            filt["source_id"] = r["source_id"]
        elif r.get("url"):
            filt["url"] = r["url"]
        elif r.get("source_id"):
            filt["source_id"] = r["source_id"]
        else:
            filt["_content_hash"] = _content_hash(r)
        ops.append(UpdateOne(filt, {"$set": r}, upsert=True))
    if ops:
        res = col.bulk_write(ops, ordered=False)
        logger.info(f"[MONGO] insérés={res.upserted_count} | maj={res.modified_count}")

def load_raw_sources(source_filter=None, limit=0) -> list[dict]:
    query = {"source": {"$in": source_filter}} if source_filter else {}
    cur   = get_db()["raw_sources"].find(query)
    if limit:
        cur = cur.limit(limit)
    return list(cur)

def load_already_normalised(source_filter=None) -> set[tuple]:
    query = {"source": {"$in": source_filter}} if source_filter else {}
    done  = set()
    for d in get_db()["normalised_sources"].find(query, {"drug_name": 1, "source": 1, "url": 1, "source_id": 1}):
        done.add((norm_drug(d.get("drug_name", "")), d.get("source", ""), d.get("url") or d.get("source_id") or ""))
    return done

def _doc_key(raw: dict) -> tuple:
    src_id = str(raw.get("source_id") or raw.get("nct_id") or raw.get("pubmed_id") or raw.get("report_id") or "")
    return (norm_drug(raw.get("drug_name", "")), (raw.get("source") or "").lower().strip(), raw.get("url") or src_id)

def is_already_normalised(raw: dict) -> bool:
    drug   = norm_drug(raw.get("drug_name", ""))
    source = (raw.get("source") or "").lower().strip()
    url    = raw.get("url") or ""
    src_id = str(raw.get("source_id") or raw.get("nct_id") or raw.get("pubmed_id") or raw.get("report_id") or "")
    filt   = {"drug_name": drug, "source": source}
    if url:
        filt["url"] = url
    elif src_id:
        filt["source_id"] = src_id
    return get_db()["normalised_sources"].count_documents(filt, limit=1) > 0

_HITS_SCORES: dict[str, HITSScore] | None = None

def _get_hits_scores() -> dict[str, HITSScore]:
    global _HITS_SCORES
    if _HITS_SCORES is None:
        _HITS_SCORES = run_hits_pipeline()
    return _HITS_SCORES

def _authority_for_source(source: str) -> tuple[float, str]:
    hit = get_source_score(source, _get_hits_scores())
    return (hit.authority_score, hit.source_type) if hit else (0.5, "hub")

def _load_spacy():
    HEAVY = ["parser", "tagger", "attribute_ruler", "lemmatizer"]
    for model in ("en_core_sci_sm", "en_core_web_sm"):
        try:
            m = spacy.load(model)
            for c in HEAVY:
                if c in m.pipe_names:
                    m.disable_pipe(c)
            if "sentencizer" not in m.pipe_names and "senter" not in m.pipe_names:
                m.add_pipe("sentencizer", first=True)
            return m
        except Exception:
            continue
    raise RuntimeError("Aucun modèle spaCy disponible")

_spacy_nlp = _load_spacy()

NER_LABELS = {"DISEASE", "SYMPTOM", "CONDITION", "CHEMICAL", "ENTITY"}

KNOWN_SIDE_EFFECTS_SEEDS = {
    "nausea", "vomiting", "diarrhea", "constipation", "dizziness",
    "drowsiness", "headache", "rash", "itching", "hives", "edema",
    "swelling", "palpitations", "insomnia", "nervousness", "anxiety",
    "depression", "fatigue", "weight gain", "weight loss", "bleeding",
    "bruising", "hypersensitivity", "drug hypersensitivity", "toxicity",
    "liver damage", "kidney damage", "hepatotoxicity", "nephrotoxicity",
    "allergic reaction", "anaphylaxis", "photosensitivity",
    "abdominal discomfort", "appetite loss", "blurred vision", "chest tightness",
    "dry mouth", "dyspepsia", "dyspnea", "elevated liver enzymes",
    "flatulence", "fluid retention", "flushing", "gastritis",
    "gastrointestinal bleeding", "hot flashes", "hypertension", "hypotension",
    "increased heart rate", "injection site reaction", "joint pain",
    "leukopenia", "memory impairment", "muscle cramps", "muscle weakness",
    "myalgia", "neutropenia", "oral ulcers", "orthostatic hypotension",
    "peripheral edema", "pruritus", "renal impairment", "sleep disturbance",
    "somnolence", "stomatitis", "sweating", "thrombocytopenia", "tinnitus",
    "upper abdominal pain", "urticaria", "vertigo", "xerostomia",
}

KNOWN_SYMPTOMS_SEEDS = {
    "pain", "back pain", "chest pain", "stomach pain", "migraine",
    "cramps", "inflammation", "fever", "shortness of breath",
    "weakness", "numbness", "tingling", "paresthesia", "confusion",
    "congestion", "bloating", "tremor", "hallucination",
    "pins and needles", "sleeplessness",
    "alopecia", "arthralgia", "cognitive decline", "diplopia",
    "dysphagia", "dysphonia", "dysuria", "epistaxis", "erythema",
    "hemoptysis", "hematuria", "hyperhidrosis", "hypoglycemia",
    "jaundice", "lymphadenopathy", "malaise", "pallor", "polyuria",
    "polydipsia", "ptosis", "purpura", "rhinorrhea", "rigidity",
    "seborrhea", "seizure", "syncope", "tachycardia", "urinary retention",
    "wheezing", "xerophthalmia",
}

SYMPTOM_ALIASES = {
    "pins and needles": "paresthesia", "paraesthesia": "paresthesia",
    "sleeplessness": "insomnia", "drug hypersensitivity": "hypersensitivity",
    "stomach pain or cramps": "stomach pain", "abdominal pain": "stomach pain",
    "throwing up": "vomiting", "sick to stomach": "nausea",
    "hair loss": "alopecia", "loss of hair": "alopecia",
    "high blood pressure": "hypertension", "low blood pressure": "hypotension",
    "irregular heartbeat": "palpitations", "shortness of breath": "dyspnea",
    "difficulty breathing": "dyspnea", "blurry vision": "blurred vision",
    "loss of appetite": "appetite loss", "decreased appetite": "appetite loss",
    "upset stomach": "dyspepsia", "stomach upset": "dyspepsia",
    "water retention": "fluid retention", "joint aches": "joint pain",
    "muscle pain": "myalgia", "muscle ache": "myalgia",
    "skin redness": "erythema", "fainting": "syncope", "dry eyes": "xerophthalmia",
}

NER_STOPWORDS = {
    "randomised", "randomized", "placebo", "placebo-controlled", "double-blind",
    "participants", "patients", "subjects", "investigators", "study", "trial",
    "groups", "treatment", "control", "endpoint", "primary endpoint",
    "secondary endpoints", "weeks", "days", "months", "augmentation",
    "multicentre", "prospective", "retrospective", "phase", "orally", "assigned",
    "open-label", "adaptive-design",
}

CT_EXTRA_STOPWORDS = {
    "celecoxib", "minocycline", "gabapentin", "methotrexate", "erlotinib",
    "capecitabine", "oxaliplatin", "acetaminophen", "ibuprofen", "naproxen",
    "opioid", "opioids", "nsaids", "non-steroidal anti-inflammatory drugs",
    "anti-inflammatory", "antidepressant", "chemotherapy", "radiotherapy",
    "neoadjuvant therapy", "neoadjuvant treatment", "chemoradiotherapy",
    "capox", "scrt", "lcrt", "tme", "serrac", "larc", "dfs", "ccr", "mco",
    "grade 3-4", "anal verge", "anal preservation", "basal ganglion",
    "thalamus", "hematoma", "brain edema", "clinical complete response",
    "complete response", "pathological", "neoadjuvant",
    "randomized phase ii trial", "phase iia", "phase 2",
}

CT_DISEASE_SYMPTOMS = {
    "hyperphagia", "hypersomnia", "fatigue", "weight gain", "leaden paralysis",
    "anhedonia", "insomnia", "cognitive impairment", "anorexia",
    "psychomotor retardation", "irritability", "hopelessness", "suicidal ideation",
    "postoperative pain", "chronic pain", "acute pain", "nociceptive pain",
    "neuroinflammation", "systemic inflammation", "inflammation",
}

CT_SYM_OVERRIDES = {
    "chronic pain", "postoperative pain", "acute pain", "inflammation",
    "fatigue", "anorexia", "cognitive impairment", "insomnia",
    "weight gain", "hyperphagia", "hypersomnia",
}

CONDITION_MAP = {
    "major depressive disorder": "major_depressive_disorder", "mdd": "major_depressive_disorder",
    "inflammation": "inflammation", "pain": "pain",
    "acute myeloid leukemia": "acute_myeloid_leukemia", "aml": "acute_myeloid_leukemia",
    "colds": "common_cold", "allergies": "allergy", "hay fever": "allergic_rhinitis",
    "back pain": "back_pain", "acute kidney injury": "acute_kidney_injury",
    "intracerebral hemorrhage": "intracerebral_hemorrhage",
    "locally advanced rectal cancer": "locally_advanced_rectal_cancer",
    "head and neck cancer": "head_and_neck_cancer",
    "postoperative pain": "postoperative_pain", "post operative pain": "postoperative_pain",
    "neoadjuvant therapy": "neoadjuvant_therapy", "submental fat": "submental_fat",
    "surgery": "surgery", "unknown": "unknown",
}

MESH_EXCLUDE = {
    "humans", "animals", "male", "female", "adult", "aged", "cell line, tumor",
    "hl-60 cells", "cell survival", "signal transduction",
    "membrane potential, mitochondrial", "reactive oxygen species",
}

DOSAGE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|ml|tablet|pill|capsule)s?\b", re.I)

_SE_CONTEXT_RE = re.compile(
    r"(?:side[\s-]?effects?\s+(?:include|included|such as|like|:)\s*"
    r"|adverse\s+(?:effects?|reactions?|events?)\s+(?:include|:)?\s*"
    r"|(?:may|can|could)\s+cause\s+"
    r"|(?:caused?|experienced?|reported?|developed?|noticed?)\s+(?:some\s+)?"
    r"|unwanted\s+effects?\s+(?:include|:)?\s*"
    r"|reactions?\s+(?:include|:)?\s*)", re.I,
)
_SYM_CONTEXT_RE = re.compile(
    r"(?:symptoms?\s+(?:of|include|such as|:)\s*"
    r"|signs?\s+(?:of|include|:)\s*"
    r"|(?:suffering|suffer)\s+from\s+"
    r"|(?:diagnosed|diagnosis)\s+with\s+"
    r"|(?:treating?|treatment\s+of)\s+)", re.I,
)

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def norm_drug(n: str) -> str:
    return n.strip().lower() if n else ""

def norm_condition(raw: str) -> str:
    raw = raw.strip().lower()
    if "," in raw:
        parts  = [p.strip() for p in raw.split(",")]
        mapped = [CONDITION_MAP[p] for p in parts if p in CONDITION_MAP]
        return mapped[0] if mapped else parts[0].replace(" ", "_")
    return CONDITION_MAP.get(raw, raw.replace(" ", "_"))

def dedup_symptoms(syms: list[str]) -> list[str]:
    result = []
    for s in sorted(syms, key=len, reverse=True):
        if not any(s in kept for kept in result):
            result.append(s)
    return sorted(result)

def _apply_meddra(terms: list[str]) -> list[str]:
    return [normalise_term(t) for t in terms]

def _extract_ct_symptoms(text: str) -> list[str]:
    tl = text.lower()
    return [s for s in CT_DISEASE_SYMPTOMS if re.search(r"\b" + re.escape(s) + r"\b", tl)]

def _lexical_fallback(text: str, drug_name: str, extra_stop=None) -> tuple[set, set]:
    stop      = NER_STOPWORDS | (extra_stop or set())
    tl        = text.lower()
    se_found  : set[str] = set()
    sym_found : set[str] = set()
    for seed in KNOWN_SIDE_EFFECTS_SEEDS:
        if seed not in stop and re.search(r"\b" + re.escape(seed) + r"\b", tl):
            se_found.add(SYMPTOM_ALIASES.get(seed, seed))
    for seed in KNOWN_SYMPTOMS_SEEDS:
        aliased = SYMPTOM_ALIASES.get(seed, seed)
        if seed not in stop and aliased not in se_found and re.search(r"\b" + re.escape(seed) + r"\b", tl):
            sym_found.add(aliased)
    sym_found -= se_found
    return se_found, sym_found

def _extract_context_terms(text: str) -> tuple[set, set]:
    se_ctx : set[str] = set()
    sym_ctx: set[str] = set()
    for m in _SE_CONTEXT_RE.finditer(text):
        win = text[m.end(): m.end() + 200].lower()
        for seed in KNOWN_SIDE_EFFECTS_SEEDS:
            if re.search(r"\b" + re.escape(seed) + r"\b", win):
                se_ctx.add(SYMPTOM_ALIASES.get(seed, seed))
    for m in _SYM_CONTEXT_RE.finditer(text):
        win = text[m.end(): m.end() + 200].lower()
        for seed in KNOWN_SYMPTOMS_SEEDS:
            if seed not in se_ctx and re.search(r"\b" + re.escape(seed) + r"\b", win):
                sym_ctx.add(SYMPTOM_ALIASES.get(seed, seed))
    sym_ctx -= se_ctx
    return se_ctx, sym_ctx

def classify_text(text, drug_name, source, static_se=None, static_sym=None, extra_stop=None):
    force_fallback = source.lower() in {"clinicaltrials", "openfda", "vigiaccess", "medlineplus"}
    nlp_result     = analyse_text(text=text, drug_name=drug_name, source=source)
    se_set  = set(nlp_result.side_effects)
    sym_set = set(nlp_result.symptoms)
    dos_set = set(nlp_result.dosages)

    for term in (static_se or []):
        se_set.add(SYMPTOM_ALIASES.get(term.lower(), term.lower()))
    for term in (static_sym or []):
        t = SYMPTOM_ALIASES.get(term.lower(), term.lower())
        if t not in se_set:
            sym_set.add(t)

    if text:
        stop = NER_STOPWORDS | (extra_stop or set())
        doc  = _spacy_nlp(text[:2000])
        for sent in doc.sents:
            if any(
                ent.label_ in NER_LABELS and len(ent.text.lower().strip()) > 3
                and ent.text.lower().strip() not in stop
                and not re.match(r"^[\d\-/]+$", ent.text.lower().strip())
                for ent in sent.ents
            ):
                r = analyse_text(text=sent.text.strip(), drug_name=drug_name, source=source)
                se_set  |= set(r.side_effects)
                sym_set |= set(r.symptoms)
                dos_set |= set(r.dosages)

    se_ctx, sym_ctx = _extract_context_terms(text)
    se_set  |= se_ctx
    sym_set |= sym_ctx

    if not bool(se_set or sym_set) or force_fallback:
        se_fb, sym_fb = _lexical_fallback(text, drug_name, extra_stop)
        se_set  |= se_fb
        sym_set |= sym_fb
    elif source in ("reddit", "drugs.com", "webmd"):
        se_fb, sym_fb = _lexical_fallback(text, drug_name, extra_stop)
        se_set  |= se_fb
        sym_set |= sym_fb

    stop_all = NER_STOPWORDS | (extra_stop or set())
    se_set   = {normalise_term(s) for s in se_set  if s not in stop_all and len(s) > 2}
    sym_set  = {normalise_term(s) for s in sym_set if s not in stop_all and len(s) > 2}
    sym_set -= se_set
    return dedup_symptoms(sorted(se_set)), dedup_symptoms(sorted(sym_set)), sorted(dos_set)

def _base(doc: dict, source: str) -> dict:
    auth_score, src_type = _authority_for_source(source)
    drug_name = norm_drug(doc.get("drug_name", ""))
    return {
        "drug_name": drug_name, "drug_class": _get_drug_class(drug_name),
        "source": source, "url": doc.get("url"),
        "source_type": src_type, "source_authority": auth_score,
        "symptoms": [], "side_effects": [], "conditions": [], "dosages": [], "text_clean": "",
    }

def norm_medlineplus(doc: dict) -> dict:
    out   = _base(doc, "medlineplus")
    se_raw = doc.get("side_effects", {})
    all_se = [s.lower() for s in (se_raw.get("mild") or []) + (se_raw.get("serious") or [])]
    text   = clean_text(" ".join(doc.get("prescribed_for") or []) + " " + " ".join(doc.get("description") or []))
    if not text.strip() and all_se:
        text = f"{doc.get('drug_name', '')} may cause " + ", ".join(all_se[:10])
    se, sym, dos = classify_text(text, doc.get("drug_name", ""), "medlineplus", static_se=all_se)
    out.update({"side_effects": se, "symptoms": sym, "dosages": dos, "text_clean": text[:1000]})
    return out

def norm_clinicaltrials(doc: dict) -> dict:
    out  = _base(doc, "clinicaltrials")
    text = clean_text(doc.get("summary", ""))
    conds = []
    for c in (doc.get("conditions") or []):
        n = norm_condition(c)
        if n and n not in conds:
            conds.append(n)
    se, sym, dos = classify_text(text, doc.get("drug_name", ""), "clinicaltrials",
                                 extra_stop=CT_EXTRA_STOPWORDS, static_sym=_extract_ct_symptoms(text))
    tl = text.lower()
    rescued = {
        SYMPTOM_ALIASES.get(s, s) for s in KNOWN_SIDE_EFFECTS_SEEDS
        if s not in CT_EXTRA_STOPWORDS
        and SYMPTOM_ALIASES.get(s, s) not in CT_SYM_OVERRIDES
        and SYMPTOM_ALIASES.get(s, s) not in se
        and re.search(r"\b" + re.escape(s) + r"\b", tl)
    }
    se = list(set(se) | rescued)
    ct_sym = set(_extract_ct_symptoms(text))
    se  = [e for e in se if e not in ct_sym]
    sym = dedup_symptoms(sorted(set(sym) | {s for s in ct_sym if s not in se}))
    cond_flat = {c.replace("_", " ") for c in conds}
    sym = [s for s in sym if s not in cond_flat and s.replace(" ", "_") not in set(conds)]
    se  = _apply_meddra(list(set(se)))
    sym = _apply_meddra(list(set(sym)))
    out.update({
        "side_effects": dedup_symptoms(sorted(set(se))), "symptoms": dedup_symptoms(sorted(set(sym))),
        "conditions": conds, "dosages": dos, "text_clean": text[:3000],
        "source_id": doc.get("nct_id"),
        "source_status": (doc.get("status") or "").lower() or None,
        "source_title": clean_text(doc.get("title", "")) or None,
    })
    return out

def norm_drugscom(doc: dict) -> dict:
    out  = _base(doc, "drugs.com")
    text = clean_text(doc.get("review_text", ""))
    se, sym, dos = classify_text(text, doc.get("drug_name", ""), "drugs.com")
    out.update({
        "side_effects": se, "symptoms": sym,
        "conditions": [norm_condition(doc.get("condition") or "unknown")],
        "dosages": dos, "text_clean": text[:1000],
        "source_id": str(doc.get("review_key") or doc.get("review_id") or doc.get("_id") or "") or None,
    })
    return out

def norm_openfda(doc: dict) -> dict:
    out      = _base(doc, "openfda")
    sym_raw  = clean_text(doc.get("symptom", ""))
    dosage   = (doc.get("dosage") or "").upper()
    drug     = doc.get("drug_name", "")
    ctx_text = f"{drug} caused {sym_raw}" if sym_raw else sym_raw
    se, sym, dos = classify_text(ctx_text, drug, "openfda", static_se=[sym_raw] if sym_raw else [])
    out.update({
        "side_effects": se, "symptoms": sym,
        "dosages": [] if dosage in ("UNK", "UNKNOWN", "") else [dosage.lower()],
        "text_clean": sym_raw, "source_id": str(doc.get("report_id", "")),
    })
    return out

def norm_pubmed(doc: dict) -> dict:
    out = _base(doc, "pubmed")
    abstract, mesh_terms, title_text = "", [], ""
    try:
        root       = etree.fromstring(doc.get("content", "").encode())
        abstract   = clean_text(" ".join(n.text or "" for n in root.iter("AbstractText")))
        mesh_terms = [n.text.strip().lower() for n in root.iter("DescriptorName")
                      if n.text and n.text.strip().lower() not in MESH_EXCLUDE]
        title_text = clean_text(" ".join(n.text or "" for n in root.iter("ArticleTitle")))
    except Exception as e:
        logger.warning(f"PubMed XML parse error: {e}")
    kw_disease = ("pain", "disorder", "syndrome", "disease", "injury", "cancer", "tumor", "leukemia", "hemorrhage", "inflammation", "apoptosis", "toxicity")
    mesh_relevant = [m for m in mesh_terms if any(k in m for k in kw_disease)]
    full_text = f"{title_text}. {abstract}".strip(". ")
    se, sym, dos = classify_text(full_text, doc.get("drug_name", ""), "pubmed", static_sym=mesh_relevant)
    conds = [norm_condition(m) for m in mesh_terms
             if any(k in m for k in ("leukemia", "cancer", "disorder", "disease", "injury", "syndrome", "hemorrhage"))]
    out.update({
        "side_effects": se, "symptoms": sym, "conditions": conds, "dosages": dos,
        "text_clean": abstract[:1000], "source_id": str(doc.get("pubmed_id", "")),
        "source_title": title_text[:200] or None,
    })
    return out

def norm_reddit(doc: dict) -> dict:
    out   = _base(doc, "reddit")
    text  = clean_text(doc.get("review_text", ""))
    rmeta = doc.get("reddit_meta") or {}
    se, sym, dos = classify_text(text, doc.get("drug_name", ""), "reddit")
    out.update({
        "side_effects": se, "symptoms": sym, "dosages": dos, "text_clean": text[:1000],
        "source_id": rmeta.get("post_id"),
        "subreddit": (rmeta.get("subreddit") or "").lower() or None,
    })
    return out

def norm_vigiaccess(doc: dict) -> dict:
    out       = _base(doc, "vigiaccess")
    soc_terms = sorted({clean_text(str(r.get("soc", ""))).lower() for r in (doc.get("soc_rows") or []) if r.get("soc")})
    drug      = doc.get("drug_name", "")
    ctx_text  = f"{drug} adverse effects: " + "; ".join(soc_terms) if soc_terms else ""
    se, sym, dos = classify_text(ctx_text, drug, "vigiaccess", static_se=soc_terms)
    out.update({"side_effects": se, "symptoms": sym, "dosages": dos})
    return out

def norm_webmd(doc: dict) -> dict:
    out      = _base(doc, "webmd")
    effects  = [clean_text(e).lower() for e in (doc.get("effects") or [])]
    drug     = doc.get("drug_name", "")
    eff_text = f"{drug} may cause the following side effects: " + ", ".join(effects) if effects else ""
    se, sym, dos = classify_text(eff_text, drug, "webmd")
    out.update({"side_effects": se, "symptoms": sym, "dosages": dos, "text_clean": eff_text[:1000]})
    return out

HANDLERS = {
    "medlineplus": norm_medlineplus, "clinicaltrials": norm_clinicaltrials,
    "drugs.com": norm_drugscom, "openfda": norm_openfda,
    "pubmed": norm_pubmed, "reddit": norm_reddit,
    "vigiaccess": norm_vigiaccess, "webmd": norm_webmd,
}

def normalise_doc(raw: dict) -> dict | None:
    source  = (raw.get("source") or "").lower().strip()
    handler = HANDLERS.get(source)
    if not handler:
        logger.warning(f"Source inconnue '{source}' — ignorée.")
        return None
    try:
        return handler(raw)
    except Exception as e:
        logger.error(f"Erreur normalisation [{source}]: {e}", exc_info=True)
        return None

def run_pipeline(source_filter=None, limit=0, dry_run=False, batch_size=500, skip_done=True, parallel_mode=False) -> int:
    _get_drug_class_cache()
    _get_hits_scores()
    raw_docs     = load_raw_sources(source_filter, limit)
    already_done = load_already_normalised(source_filter) if skip_done and not dry_run and not parallel_mode else set()
    batch = []
    ok = skipped = already_skipped = 0

    for raw in raw_docs:
        if skip_done and not dry_run:
            if parallel_mode:
                if is_already_normalised(raw):
                    already_skipped += 1
                    continue
            elif _doc_key(raw) in already_done:
                already_skipped += 1
                continue
        result = normalise_doc(raw)
        if result is None:
            skipped += 1
            continue
        batch.append(result)
        ok += 1
        if not dry_run and len(batch) >= batch_size:
            save_normalised(batch)
            batch.clear()

    if not dry_run and batch:
        save_normalised(batch)
    logger.info(f"Pipeline terminé — normalisés={ok} | déjà faits={already_skipped} | erreurs={skipped}")
    return ok

if __name__ == "__main__":
    import argparse
    ALL_SOURCES = ["medlineplus", "clinicaltrials", "drugs.com", "drugscom", "openfda", "pubmed", "vigiaccess", "webmd", "reddit"]
    p = argparse.ArgumentParser(description="Bio-Sentry v5.4 — Pipeline normalisation")
    p.add_argument("--source",     metavar="SOURCE", choices=ALL_SOURCES)
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
        sources, parallel_mode = ALL_SOURCES, False

    run_pipeline(
        source_filter=sources, limit=args.limit,
        dry_run=args.dry_run, skip_done=not args.force,
        batch_size=args.batch_size, parallel_mode=parallel_mode,
    )