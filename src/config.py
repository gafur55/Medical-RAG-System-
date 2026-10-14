import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_DIR = ROOT / "data" / "eval"
FIGURES_DIR = PROCESSED_DIR / "figures"

BOOKS = {
    "mohs_layers": RAW_DIR / "MOHS_MICROGRAPHIC_SURGERY_from_layers_to_reconstruction.pdf",
}

SOURCES_PATH = ROOT / "sources.yaml"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
VECTORS_PATH = PROCESSED_DIR / "vectors.npy"
VECTOR_KEYS_PATH = PROCESSED_DIR / "vector_keys.json"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
REVIEW_SHEET_PATH = EVAL_DIR / "review_sheet.md"
RESULTS_PATH = EVAL_DIR / "results.jsonl"
JUDGE_RUNS_PATH = EVAL_DIR / "judge_runs.jsonl"
REPORTS_DIR = EVAL_DIR / "reports"

# The textbook. Chunks written before the multi-source index have no
# "source" field; they belong to this one. questions.json gold_pages are
# pages of this source too.
DEFAULT_SOURCE = "mohs_layers"

# ingestion
MIN_CHUNK_CHARS = 100
MIN_FIGURE_PIXELS = 150

# chunking for every source except the textbook (see chunking.py). The
# textbook keeps its layout blocks, so its IDs and vectors do not change.
CHUNK_TARGET_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
CHUNK_MIN_CHARS = 50
# New sources embed "Section > Subsection\n<text>" instead of the bare text,
# so a chunk keeps the meaning its heading gave it. The textbook has no
# section path, so its embedded text - and its vectors - are unchanged.
EMBED_SECTION_HEADINGS = True

# retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
USE_QUERY_PREFIX = True
RETRIEVE_KINDS = ("prose", "caption")
RETRIEVE_DOMAINS = ("clinical", "operations")   # sources.yaml "domain"
TOP_K = 5
REVIEW_POOL = 25
# Experimental knobs (see docs/retrieve.md). The defaults reproduce the
# baseline exactly - change one at a time and re-run src.score / src.sweep.
EXCLUDE_FRONT_MATTER = False   # drop chapter abstracts, keyword lists, TOC/index pages
MAX_PER_PAGE = None            # cap on chunks from one page in the top k; None = no cap
HYBRID_WEIGHT = 0.0            # 0 = embeddings only; >0 mixes in BM25 keyword scores

# generation
LLM_MODEL = "openai/gpt-oss-120b"   # Groq model id, read by generate.py
# "page":     cite as (p. 384). The original prompt; fine while the index
#             holds one paginated source.
# "numbered": cite excerpts as [2]. Needed once several sources are indexed,
#             since a page number alone no longer says which document.
CITATION_STYLE = "page"

# LLM judge (src/judge.py). Same model as the answers by default; a different
# model family judges with less self-preference, so try one when you can.
JUDGE_MODEL = LLM_MODEL


def index_manifest():
    """manifest.json written by ingest.py, or {} if there is none yet.

    Holds the index ID and one entry per indexed source (id, label, title,
    domain, type, license, url, year, n_chunks). The app reads source labels
    from here, so it never needs sources.yaml or PyYAML.
    """
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except (OSError, ValueError):
        return {}


def index_id():
    """Short ID of the current index build (written by ingest.py), or None."""
    return index_manifest().get("index_id")


def source_labels():
    """{source_id: short label} for every indexed source."""
    labels = {s["id"]: s.get("label") or s["id"]
              for s in index_manifest().get("sources", [])}
    labels.setdefault(DEFAULT_SOURCE, "Mohs textbook")
    return labels


def run_config():
    return {
        "embedding_model": EMBEDDING_MODEL,
        "use_query_prefix": USE_QUERY_PREFIX,
        "retrieve_kinds": list(RETRIEVE_KINDS),
        "retrieve_domains": list(RETRIEVE_DOMAINS),
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "top_k": TOP_K,
        "exclude_front_matter": EXCLUDE_FRONT_MATTER,
        "max_per_page": MAX_PER_PAGE,
        "hybrid_weight": HYBRID_WEIGHT,
        "llm_model": LLM_MODEL,
        "citation_style": CITATION_STYLE,
        "index_id": index_id(),
    }