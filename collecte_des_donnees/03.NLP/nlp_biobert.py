import re
import logging
import os
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
import torch
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("biosentry.nlp")

BIOBERT_NER_MODEL   = "fran-martinez/scibert-ner-medical"
FALLBACK_NER_MODEL  = "allenai/scibert_scivocab_cased"
BIOBERT_EMBED_MODEL = "dmis-lab/biobert-base-cased-v1.2"

DEVICE     = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[NLP] Device: {'GPU' if DEVICE == 0 else 'CPU'}")

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB  = "biosentry_db"
MONGO_COL = "normalised_sources_v3"

UGC_SOURCES  = {"reddit", "drugs.com", "drugcom", "drugscom"}
RelationType = Literal["side_effect", "symptom", "dosage", "drug", "unknown"]

DOSAGE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|ml|tablet|pill|capsule)s?\b", re.I)

SIDE_EFFECT_PATTERNS = [
    r"\bcaus(?:es?|ed|ing)\b", r"\binduc(?:es?|ed|ing)\b", r"\bsid[e\s]+effect",
    r"\badverse\s+(?:event|reaction|effect)", r"\bdue to\b",
    r"\bafter (?:taking|using|treatment with|administration)\b",
    r"\bfollowing (?:administration|treatment|dose)\b", r"\bsecondary to\b",
    r"\btoxicit", r"\bcomplication", r"\bundesirable\b", r"\bdrug[\s\-]induced",
    r"\bmedication[\s\-]related", r"\btreatment[\s\-]related", r"\breaction to\b",
]

SYMPTOM_PATTERNS = [
    r"\bpresent(?:s|ed|ing)?\s+with\b", r"\bmanifest(?:s|ed|ation)?\b",
    r"\bsymptom(?:s|atic)?\b", r"\bsign(?:s)?\s+of\b",
    r"\bassociat(?:ed|ion)\s+with\s+(?:the\s+)?(?:disease|condition|disorder)\b",
    r"\bunderl(?:ying|ies)\b", r"\bclinical\s+(?:feature|presentation|finding)\b",
    r"\bpatholog", r"\bdiagnos(?:ed|is)\s+with\b", r"\bcomorbid",
    r"\bcourse of (?:the\s+)?disease\b",
]

KNOWN_SIDE_EFFECTS = {
    "nausea","vomiting","diarrhea","constipation","dizziness","drowsiness","headache",
    "rash","itching","hives","edema","swelling","palpitations","insomnia","nervousness",
    "anxiety","depression","fatigue","weight gain","weight loss","bleeding","bruising",
    "hypersensitivity","drug hypersensitivity","toxicity","liver damage","kidney damage",
    "hepatotoxicity","nephrotoxicity","allergic reaction","anaphylaxis","photosensitivity",
    "abdominal discomfort","appetite loss","blurred vision","chest tightness","dry mouth",
    "dyspepsia","dyspnea","elevated liver enzymes","flatulence","fluid retention",
    "flushing","gastritis","gastrointestinal bleeding","hot flashes","hypertension",
    "hypotension","increased heart rate","injection site reaction","joint pain",
    "leukopenia","memory impairment","muscle cramps","muscle weakness","myalgia",
    "neutropenia","oral ulcers","orthostatic hypotension","peripheral edema","pruritus",
    "renal impairment","sleep disturbance","somnolence","stomatitis","sweating",
    "thrombocytopenia","tinnitus","upper abdominal pain","urticaria","vertigo","xerostomia",
}

KNOWN_SYMPTOMS = {
    "pain","back pain","chest pain","stomach pain","migraine","cramps","inflammation",
    "fever","shortness of breath","weakness","numbness","tingling","paresthesia",
    "confusion","congestion","bloating","tremor","hallucination","pins and needles",
    "sleeplessness","alopecia","arthralgia","cognitive decline","diplopia","dysphagia",
    "dysphonia","dysuria","epistaxis","erythema","hemoptysis","hematuria",
    "hyperhidrosis","hypoglycemia","jaundice","lymphadenopathy","malaise","myalgia",
    "pallor","polyuria","polydipsia","ptosis","purpura","rhinorrhea","rigidity",
    "seborrhea","seizure","syncope","tachycardia","urinary retention","wheezing","xerophthalmia",
}

SYMPTOM_ALIASES = {
    "pins and needles":"paresthesia","paraesthesia":"paresthesia",
    "sleeplessness":"insomnia","drug hypersensitivity":"hypersensitivity",
    "stomach pain or cramps":"stomach pain","abdominal pain":"stomach pain",
    "throwing up":"vomiting","sick to stomach":"nausea","hair loss":"alopecia",
    "loss of hair":"alopecia","high blood pressure":"hypertension",
    "low blood pressure":"hypotension","irregular heartbeat":"palpitations",
    "shortness of breath":"dyspnea","difficulty breathing":"dyspnea",
    "blurry vision":"blurred vision","loss of appetite":"appetite loss",
    "decreased appetite":"appetite loss","upset stomach":"dyspepsia",
    "stomach upset":"dyspepsia","water retention":"fluid retention",
    "joint aches":"joint pain","muscle pain":"myalgia","muscle ache":"myalgia",
    "skin redness":"erythema","fainting":"syncope","dry eyes":"xerophthalmia",
}

MEDDRA_NORMALISATION: dict[str, str] = {
    "nausea and vomiting":"nausea","nausea/vomiting":"nausea","gi upset":"dyspepsia",
    "gi distress":"dyspepsia","stomach ache":"stomach pain","abdominal pain":"stomach pain",
    "abdominal cramps":"stomach pain","loose stools":"diarrhea","loose bowels":"diarrhea",
    "stomach discomfort":"abdominal discomfort","indigestion":"dyspepsia",
    "heartburn":"dyspepsia","reflux":"dyspepsia","lightheadedness":"dizziness",
    "vertigo":"dizziness","dizziness or lightheadedness":"dizziness",
    "pins and needles":"paresthesia","paraesthesia":"paresthesia",
    "numbness and tingling":"paresthesia","tingling sensation":"paresthesia",
    "brain fog":"memory impairment","confusion":"memory impairment",
    "forgetfulness":"memory impairment","difficulty concentrating":"memory impairment",
    "sleeplessness":"insomnia","difficulty sleeping":"insomnia",
    "poor sleep":"sleep disturbance","sleep problems":"sleep disturbance",
    "drowsiness":"somnolence","sedation":"somnolence","feeling tired":"fatigue",
    "tiredness":"fatigue","lethargy":"fatigue","malaise":"fatigue","trembling":"tremor",
    "irregular heartbeat":"palpitations","heart palpitations":"palpitations",
    "racing heart":"palpitations","high blood pressure":"hypertension",
    "elevated blood pressure":"hypertension","low blood pressure":"hypotension",
    "tachycardia":"increased heart rate","fast heart rate":"increased heart rate",
    "skin rash":"rash","hives":"urticaria","itch":"pruritus","itching":"pruritus",
    "skin itching":"pruritus","swelling":"edema","ankle swelling":"peripheral edema",
    "leg swelling":"peripheral edema","water retention":"fluid retention",
    "puffiness":"edema","hair loss":"alopecia","loss of hair":"alopecia",
    "dry eyes":"xerophthalmia","blurry vision":"blurred vision","vision changes":"blurred vision",
    "shortness of breath":"dyspnea","difficulty breathing":"dyspnea",
    "breathlessness":"dyspnea","wheezing":"dyspnea","runny nose":"rhinorrhea",
    "nasal discharge":"rhinorrhea","muscle pain":"myalgia","muscle ache":"myalgia",
    "muscle aches":"myalgia","joint aches":"joint pain","joint ache":"joint pain",
    "arthralgia":"joint pain","bone pain":"pain","body aches":"myalgia",
    "weight increase":"weight gain","gaining weight":"weight gain",
    "weight decrease":"weight loss","losing weight":"weight loss",
    "loss of appetite":"appetite loss","decreased appetite":"appetite loss",
    "reduced appetite":"appetite loss","increased thirst":"polydipsia",
    "excessive thirst":"polydipsia","frequent urination":"polyuria",
    "excessive urination":"polyuria","liver problems":"hepatotoxicity",
    "liver injury":"hepatotoxicity","elevated liver enzymes":"hepatotoxicity",
    "kidney problems":"nephrotoxicity","kidney injury":"nephrotoxicity",
    "renal failure":"renal impairment","renal insufficiency":"renal impairment",
    "low white blood cells":"leukopenia","low platelets":"thrombocytopenia",
    "bruising easily":"bruising","unusual bleeding":"bleeding","hot flush":"hot flashes",
    "hot flushes":"hot flashes","night sweats":"sweating","excessive sweating":"sweating",
    "feeling depressed":"depression","mood changes":"depression",
    "anxiety attacks":"anxiety","feeling anxious":"anxiety","nervousness":"nervousness",
    "jitteriness":"nervousness","restlessness":"nervousness","fainting":"syncope",
    "fainting spells":"syncope","dry mouth":"xerostomia","xerostomia":"xerostomia",
    "mouth sores":"oral ulcers","mouth ulcers":"stomatitis","sore throat":"stomatitis",
    "ringing in ears":"tinnitus","ear ringing":"tinnitus",
}


def normalise_term(raw: str) -> str:
    t = raw.strip().lower()
    t = MEDDRA_NORMALISATION.get(t, t)
    return SYMPTOM_ALIASES.get(t, t)


@dataclass
class NLPResult:
    side_effects : list[str] = field(default_factory=list)
    symptoms     : list[str] = field(default_factory=list)
    dosages      : list[str] = field(default_factory=list)
    drugs        : list[str] = field(default_factory=list)
    sentiment    : dict | None = None


class BioNERExtractor:
    def __init__(self):
        self._ner = None
        for model_id in [BIOBERT_NER_MODEL, FALLBACK_NER_MODEL]:
            try:
                self._ner = hf_pipeline("ner", model=model_id, tokenizer=model_id,
                                        aggregation_strategy="simple", device=DEVICE)
                logger.info(f"[NER] Loaded: {model_id} ✓")
                return
            except Exception as e:
                logger.warning(f"[NER] Failed {model_id}: {e}")
        logger.error("[NER] No model available — seed-only mode")

    def extract(self, text: str) -> list[dict]:
        if not self._ner or not text: return []
        tokenizer = self._ner.tokenizer
        results   = []
        for chunk, offset in _chunk_text(text):
            try:
                ids   = tokenizer.encode(chunk, add_special_tokens=True, max_length=512, truncation=True)
                chunk = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            except Exception as e:
                logger.warning(f"[NER] Tokenisation error: {e}")
            try:
                for ent in self._ner(chunk):
                    label = _normalise_label(ent.get("entity_group", ""))
                    if label == "OTHER": continue
                    results.append({"text": ent["word"].strip().lower(), "label": label,
                                    "start": ent["start"] + offset, "end": ent["end"] + offset,
                                    "score": round(float(ent.get("score", 0.0)), 3)})
            except Exception as e:
                logger.warning(f"[NER] Chunk error: {e}")
        return _dedup_entities(results)


class RelationClassifier:
    def __init__(self):
        self._tokenizer = self._model = self._clf = None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(BIOBERT_EMBED_MODEL)
            self._model     = AutoModel.from_pretrained(BIOBERT_EMBED_MODEL)
            self._model.eval()
            self._model.to(DEVICE_STR)
            logger.info("[REL] BioBERT embeddings loaded ✓")
        except Exception as e:
            logger.error(f"[REL] Cannot load BioBERT: {e}")
        self._train()

    def _embed(self, text: str) -> np.ndarray | None:
        if not self._tokenizer or not self._model: return None
        try:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                     max_length=256, padding=True).to(DEVICE_STR)
            with torch.no_grad():
                out = self._model(**inputs)
            return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        except Exception as e:
            logger.warning(f"[REL] Embedding error: {e}")
            return None

    def _train(self):
        if not self._tokenizer: return
        se_templates = [
            "After taking {drug}, the patient experienced {entity}.",
            "The drug {drug} induced {entity} as a side effect.",
            "{entity} was reported as an adverse event caused by {drug}.",
            "Treatment with {drug} led to {entity}.",
            "{drug}-induced {entity} was observed in clinical trials.",
            "Side effects of {drug} include {entity}.",
            "Drug-related {entity} occurred following administration of {drug}.",
            "{entity} developed after the patient started {drug} therapy.",
            "The adverse reaction {entity} was attributed to {drug}.",
            "{drug} was discontinued due to {entity}.",
        ]
        sym_templates = [
            "The patient presented with {entity} as a manifestation of the underlying disease.",
            "{entity} is a clinical feature of the disorder being treated.",
            "Symptoms of the disease include {entity}.",
            "The patient was diagnosed with a condition characterized by {entity}.",
            "{entity} is a pathological sign of the underlying condition.",
            "Clinical presentation of the illness includes {entity}.",
            "The disease course was marked by progressive {entity}.",
            "{entity} was present before any treatment was initiated.",
            "The pre-existing condition caused {entity} in the patient.",
            "Physical examination revealed {entity} consistent with the diagnosis.",
            "The patient complained of {entity} related to their chronic illness.",
            "As the disease progressed, {entity} became more prominent.",
        ]
        drugs   = ["ibuprofen","aspirin","metformin","lisinopril","atorvastatin"]
        se_list = list(KNOWN_SIDE_EFFECTS)[:18]
        sy_list = list(KNOWN_SYMPTOMS)[:15]
        X, y = [], []
        for tmpl in se_templates:
            for ent in se_list:
                emb = self._embed(tmpl.format(drug=drugs[hash(ent) % len(drugs)], entity=ent))
                if emb is not None: X.append(emb); y.append(0)
        for tmpl in sym_templates:
            for ent in sy_list:
                emb = self._embed(tmpl.format(entity=ent))
                if emb is not None: X.append(emb); y.append(1)
        if len(X) < 10:
            logger.warning("[REL] Not enough examples — classifier disabled")
            return
        self._clf = LogisticRegression(max_iter=500, C=0.8)
        self._clf.fit(np.array(X), np.array(y))
        logger.info(f"[REL] Classifier trained on {len(X)} silver examples ✓")

    def classify(self, entity_text: str, context: str, drug_name: str) -> RelationType:
        if self._clf is not None:
            emb = self._embed(f"{drug_name} context: {context}" if drug_name else context)
            if emb is not None:
                pred  = self._clf.predict([emb])[0]
                proba = self._clf.predict_proba([emb])[0]
                if max(proba) >= 0.82:
                    return "side_effect" if pred == 0 else "symptom"
        return _classify_by_patterns(entity_text, context, drug_name)


class SentimentAnalyser:
    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def analyse(self, text: str) -> dict:
        if not text: return {"label": "neutral", "score": 0.0}
        c = self._vader.polarity_scores(text)["compound"]
        return {"label": "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral",
                "score": round(c, 4)}


_ner_extractor      : BioNERExtractor    | None = None
_relation_clf       : RelationClassifier | None = None
_sentiment_analyser : SentimentAnalyser  | None = None

def _get_ner()       -> BioNERExtractor:
    global _ner_extractor
    if _ner_extractor is None: _ner_extractor = BioNERExtractor()
    return _ner_extractor

def _get_rel_clf()   -> RelationClassifier:
    global _relation_clf
    if _relation_clf is None: _relation_clf = RelationClassifier()
    return _relation_clf

def _get_sentiment() -> SentimentAnalyser:
    global _sentiment_analyser
    if _sentiment_analyser is None: _sentiment_analyser = SentimentAnalyser()
    return _sentiment_analyser


def analyse_text(text: str, drug_name: str = "", source: str = "") -> NLPResult:
    if not text: return NLPResult()
    text_lower   = text.lower()
    source_lower = source.lower()

    seed_se  = {SYMPTOM_ALIASES.get(s, s) for s in KNOWN_SIDE_EFFECTS
                if re.search(r'\b' + re.escape(s) + r'\b', text_lower)}
    seed_sym = {SYMPTOM_ALIASES.get(s, s) for s in KNOWN_SYMPTOMS
                if re.search(r'\b' + re.escape(s) + r'\b', text_lower)}

    side_effects: set[str] = set()
    symptoms    : set[str] = set()
    dosages     : set[str] = {f"{m.group(1)} {m.group(2).lower()}" for m in DOSAGE_RE.finditer(text)}
    drugs       : set[str] = set()

    entities        = _get_ner().extract(text)
    rel_clf         = _get_rel_clf()
    clf_covered     : set[str] = set()

    for ent in entities:
        raw   = ent["text"]
        token = normalise_term(raw)
        if ent["label"] == "DRUG":
            if token != (drug_name or "").lower(): drugs.add(token)
        elif ent["label"] == "DOSAGE":
            dosages.add(token)
        elif ent["label"] in ("DISEASE", "SYMPTOM"):
            pos     = ent["start"]
            context = text[max(0, pos - 200): min(len(text), ent["end"] + 200)]
            relation = rel_clf.classify(raw, context, drug_name)
            clf_covered.add(token)
            if relation == "side_effect":
                side_effects.add(token); symptoms.discard(token)
            elif relation == "symptom":
                if token not in side_effects: symptoms.add(token)
            else:
                if raw in KNOWN_SIDE_EFFECTS or token in KNOWN_SIDE_EFFECTS:
                    side_effects.add(token)
                elif raw in KNOWN_SYMPTOMS or token in KNOWN_SYMPTOMS:
                    symptoms.add(token)

    for s in seed_se:
        if s not in clf_covered: side_effects.add(s)
    for s in seed_sym:
        if s not in clf_covered and s not in side_effects: symptoms.add(s)

    sentiment = _get_sentiment().analyse(text) if any(s in source_lower for s in UGC_SOURCES) else None

    side_effects = {normalise_term(s) for s in side_effects}
    symptoms     = {normalise_term(s) for s in symptoms} - side_effects

    return NLPResult(side_effects=sorted(side_effects), symptoms=sorted(symptoms),
                     dosages=sorted(dosages), drugs=sorted(drugs), sentiment=sentiment)


def run_nlp_pipeline(mongo_uri=MONGO_URI, db_name=MONGO_DB, col_name=MONGO_COL, batch_size=50) -> dict:
    from pymongo import UpdateOne
    stats = {"processed": 0, "updated": 0, "errors": 0}
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        col    = client[db_name][col_name]
    except Exception as e:
        logger.error(f"[MONGO] Connection failed: {e}"); return stats

    query    = {"text_clean": {"$exists": True, "$ne": ""}}
    total    = col.count_documents(query)
    cursor   = col.find(query, {"_id": 1, "text_clean": 1, "drug_name": 1, "source": 1})
    bulk_ops = []

    for doc in cursor:
        stats["processed"] += 1
        try:
            r      = analyse_text(doc.get("text_clean",""), doc.get("drug_name",""), doc.get("source",""))
            fields = {"side_effects": r.side_effects, "symptoms": r.symptoms, "dosages": r.dosages}
            if r.sentiment: fields["sentiment"] = r.sentiment
            bulk_ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": fields}))
            if len(bulk_ops) >= batch_size:
                stats["updated"] += col.bulk_write(bulk_ops).modified_count
                bulk_ops = []
                logger.info(f"[MONGO] Processed: {stats['processed']}/{total}")
        except Exception as e:
            logger.warning(f"[MONGO] Error doc {doc.get('_id')}: {e}")
            stats["errors"] += 1

    if bulk_ops: stats["updated"] += col.bulk_write(bulk_ops).modified_count
    client.close()
    logger.info(f"[MONGO] NLP pipeline done: {stats}")
    return stats


def _classify_by_patterns(entity_text: str, context: str, drug_name: str) -> RelationType:
    ctx   = context.lower()
    drug  = (drug_name or "").lower()
    se_sc = sum(1 for p in SIDE_EFFECT_PATTERNS if re.search(p, ctx))
    sy_sc = sum(1 for p in SYMPTOM_PATTERNS     if re.search(p, ctx))
    if drug and drug in ctx: se_sc += 2
    if se_sc == 0 and sy_sc == 0: return "unknown"
    return "side_effect" if se_sc >= sy_sc else "symptom"

def _normalise_label(raw: str) -> str:
    raw = raw.upper()
    if any(k in raw for k in ("CHEM","DRUG","MED","COMPOUND","MOLECULE")):   return "DRUG"
    if any(k in raw for k in ("DIS","DISEASE","CONDITION","DISORDER","SYNDROME")): return "DISEASE"
    if any(k in raw for k in ("SYMPTOM","SIGN","FINDING","MANIFESTATION")):  return "SYMPTOM"
    if any(k in raw for k in ("DOSE","DOSAGE","QUANTITY","STRENGTH")):       return "DOSAGE"
    return "OTHER"

def _chunk_text(text: str, max_words: int = 150) -> list[tuple[str, int]]:
    words = text.split()
    return [(" ".join(words[i:i+max_words]),
             len(" ".join(words[:i])) + (1 if i > 0 else 0))
            for i in range(0, len(words), max_words)]

def _dedup_entities(entities: list[dict]) -> list[dict]:
    seen, result = set(), []
    for e in entities:
        key = (e["text"], e["label"])
        if key not in seen and len(e["text"]) > 2:
            seen.add(key); result.append(e)
    return result


if __name__ == "__main__":
    t = {
        "drug": "celecoxib", "source": "clinicaltrials",
        "text": (
            "This trial compares two analgesic regimens. "
            "Non-Steroidal Anti-Inflammatory Drugs plus acetaminophen or low dose opioids. "
            "The key question is which option will have the best outcomes "
            "and with the fewest side effects? Patients may experience nausea, "
            "vomiting, dizziness after taking celecoxib. The underlying disease "
            "caused postoperative pain and inflammation."
        ),
    }
    r = analyse_text(t["text"], t["drug"], t["source"])
    print(f"Drug: {t['drug']}  |  Source: {t['source']}")
    print(f"Side effects : {r.side_effects}")
    print(f"Symptoms     : {r.symptoms}")
    print(f"Dosages      : {r.dosages}")
