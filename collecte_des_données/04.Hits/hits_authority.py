import logging
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("biosentry.hits")

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "biosentry"

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB  = "biosentry_db"
MONGO_COL = "normalised_sources_v3"

ES_HOST  = "http://localhost:9200"
ES_INDEX = "biosentry_index"

HITS_MAX_ITER       = 100
HITS_TOL            = 1e-6
AUTHORITY_THRESHOLD = 0.5

CITATION_EDGES: list[tuple[str, str]] = [
    ("medlineplus", "pubmed"),
    ("medlineplus", "clinicaltrials"),
    ("medlineplus", "openfda"),
    ("webmd",       "pubmed"),
    ("webmd",       "clinicaltrials"),
    ("webmd",       "openfda"),
    ("webmd",       "vigiaccess"),
    ("webmd",       "medlineplus"),
    ("clinicaltrials", "pubmed"),
    ("openfda",     "pubmed"),
    ("openfda",     "clinicaltrials"),
    ("vigiaccess",  "pubmed"),
    ("vigiaccess",  "openfda"),
    ("drugscom",    "webmd"),
    ("drugscom",    "medlineplus"),
    ("drugscom",    "openfda"),
    ("reddit",      "webmd"),
    ("reddit",      "drugscom"),
]

SOURCE_LABELS: dict[str, str] = {
    "pubmed"        : "PubMed",
    "clinicaltrials": "ClinicalTrials.gov",
    "openfda"       : "OpenFDA (FAERS)",
    "vigiaccess"    : "VigiAccess (OMS)",
    "medlineplus"   : "MedlinePlus",
    "webmd"         : "WebMD",
    "drugscom"      : "Drugs.com",
    "reddit"        : "Reddit",
}


@dataclass
class HITSScore:
    source          : str
    label           : str
    authority_score : float
    hub_score       : float
    source_type     : str


class HITSComputer:
    def __init__(self, edges: list[tuple[str, str]]):
        self.edges   = edges
        self.sources = sorted(set(s for edge in edges for s in edge))
        self.idx     = {s: i for i, s in enumerate(self.sources)}
        self.n       = len(self.sources)
        self.A       = np.zeros((self.n, self.n), dtype=np.float64)
        for hub, auth in edges:
            if hub in self.idx and auth in self.idx:
                self.A[self.idx[hub]][self.idx[auth]] = 1.0
        logger.info(f"[HITS] Adjacency matrix: {self.n}×{self.n}, {int(self.A.sum())} edges")

    def compute(self, max_iter=HITS_MAX_ITER, tol=HITS_TOL) -> dict[str, "HITSScore"]:
        authority = np.ones(self.n, dtype=np.float64)
        hub       = np.ones(self.n, dtype=np.float64)

        for i in range(max_iter):
            new_auth = self.A.T @ hub
            new_hub  = self.A   @ new_auth
            norm_a, norm_h = np.linalg.norm(new_auth), np.linalg.norm(new_hub)
            if norm_a > 0: new_auth /= norm_a
            if norm_h > 0: new_hub  /= norm_h
            delta = np.linalg.norm(new_auth - authority) + np.linalg.norm(new_hub - hub)
            authority, hub = new_auth, new_hub
            if delta < tol:
                logger.info(f"[HITS] Converged at iteration {i+1} (Δ={delta:.2e})")
                break
        else:
            logger.warning(f"[HITS] Did not converge after {max_iter} iterations")

        if authority.max() > 0: authority /= authority.max()
        if hub.max() > 0:       hub       /= hub.max()

        results: dict[str, HITSScore] = {}
        for source in self.sources:
            i   = self.idx[source]
            a   = round(float(authority[i]), 4)
            h   = round(float(hub[i]),       4)
            results[source] = HITSScore(
                source          = source,
                label           = SOURCE_LABELS.get(source, source),
                authority_score = a,
                hub_score       = h,
                source_type     = "authority" if a >= AUTHORITY_THRESHOLD else "hub",
            )

        sorted_res = sorted(results.values(), key=lambda x: x.authority_score, reverse=True)
        print(f"  {'Source':<22} {'Type':<12} {'Authority':>10} {'Hub':>8}")
        print("  " + "─" * 56)
        for r in sorted_res:
            print(f"  {r.label:<22} {r.source_type:<12} {r.authority_score:>10.4f} {r.hub_score:>8.4f}")
        return results


class Neo4jManager:
    def __init__(self, uri, user, password):
        self._driver = None
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            logger.info("[NEO4J] Connected ✓")
        except Exception as e:
            logger.error(f"[NEO4J] Connection failed: {e}")

    def close(self):
        if self._driver: self._driver.close()

    def build_graph(self, edges):
        if not self._driver: return
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE")
            for source in set(n for e in edges for n in e):
                s.run("MERGE (s:Source {name:$n}) SET s.label=$l",
                      n=source, l=SOURCE_LABELS.get(source, source))
            for hub, auth in edges:
                s.run("MATCH (h:Source{name:$h}),(a:Source{name:$a}) MERGE (h)-[:CITE]->(a)",
                      h=hub, a=auth)
        logger.info(f"[NEO4J] Graph built: {len(set(n for e in edges for n in e))} nodes, {len(edges)} edges")

    def store_hits_scores(self, scores: dict[str, HITSScore]):
        if not self._driver: return
        with self._driver.session() as s:
            for source, sc in scores.items():
                s.run("MATCH (s:Source{name:$n}) SET s.authority_score=$a, s.hub_score=$h, s.source_type=$t",
                      n=source, a=sc.authority_score, h=sc.hub_score, t=sc.source_type)
        logger.info(f"[NEO4J] HITS scores stored for {len(scores)} nodes ✓")


def _update_mongodb(scores: dict[str, HITSScore]) -> int:
    SOURCE_VARIANTS = {
        "pubmed"        : ["pubmed"],
        "clinicaltrials": ["clinicaltrials", "clinicaltrials.gov", "ct.gov"],
        "openfda"       : ["openfda", "fda", "faers"],
        "vigiaccess"    : ["vigiaccess", "who", "oms"],
        "medlineplus"   : ["medlineplus", "nlm"],
        "webmd"         : ["webmd"],
        "drugscom"      : ["drugs.com", "drugscom"],
        "reddit"        : ["reddit"],
    }
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        col    = client[MONGO_DB][MONGO_COL]
    except Exception as e:
        logger.error(f"[MONGO] Connection failed: {e}")
        return 0

    total = 0
    for key, score in scores.items():
        res = col.update_many(
            {"source": {"$in": SOURCE_VARIANTS.get(key, [key])}},
            {"$set": {"source_authority": score.authority_score, "source_type": score.source_type}}
        )
        total += res.modified_count
    client.close()
    logger.info(f"[MONGO] Updated {total} documents")
    return total


def _update_elasticsearch(scores: dict[str, HITSScore]) -> bool:
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(ES_HOST)
        if not es.ping():
            logger.warning("[ES] Not reachable")
            return False
    except Exception as e:
        logger.warning(f"[ES] Failed: {e}")
        return False

    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(index=ES_INDEX, body={"mappings": {"properties": {
            "drug_name": {"type":"keyword"}, "source": {"type":"keyword"},
            "source_type": {"type":"keyword"}, "source_authority": {"type":"float"},
            "symptoms": {"type":"text","analyzer":"english"},
            "side_effects": {"type":"text","analyzer":"english"},
            "text_clean": {"type":"text","analyzer":"english"},
            "confidence": {"type":"keyword"}, "hub_score": {"type":"float"},
        }}})

    for key, sc in scores.items():
        es.index(index=ES_INDEX, id=f"hits_score_{key}", body={
            "source": key, "source_type": sc.source_type,
            "source_authority": sc.authority_score, "hub_score": sc.hub_score, "label": sc.label,
        })
    logger.info("[ES] HITS scores indexed ✓")
    return True


def run_hits_pipeline(
    neo4j_uri=NEO4J_URI, neo4j_user=NEO4J_USER, neo4j_password=NEO4J_PASSWORD,
    mongo_uri=MONGO_URI, update_mongo=True, update_es=False,
) -> dict[str, HITSScore]:
    logger.info("[HITS] Starting pipeline")
    scores = HITSComputer(CITATION_EDGES).compute()
    neo4j  = Neo4jManager(neo4j_uri, neo4j_user, neo4j_password)
    neo4j.build_graph(CITATION_EDGES)
    neo4j.store_hits_scores(scores)
    neo4j.close()
    if update_mongo: _update_mongodb(scores)
    if update_es:    _update_elasticsearch(scores)
    logger.info("[HITS] Pipeline done ✓")
    return scores


def get_source_score(source: str, scores: dict[str, HITSScore]) -> HITSScore | None:
    clean = source.lower().replace(".", "").replace("-", "").replace(" ", "")
    if clean in scores: return scores[clean]
    for key in scores:
        if key in clean or clean in key: return scores[key]
    return None


if __name__ == "__main__":
    scores = HITSComputer(CITATION_EDGES).compute()
    for src in ["pubmed","clinicaltrials","openfda","vigiaccess","medlineplus","webmd","drugscom","reddit"]:
        if src in scores:
            sc = scores[src]
            print(f"  {SOURCE_LABELS[src]:<22} authority={sc.authority_score:.4f}  type={sc.source_type}")
    print("\n  get_source_score('drugs.com') :", get_source_score("drugs.com", scores))
    print("  get_source_score('reddit') :",     get_source_score("reddit", scores))
