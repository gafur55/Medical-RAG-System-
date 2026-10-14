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

# ingestion
MIN_CHUNK_CHARS = 100
MIN_FIGURE_PIXELS = 150

# chunking for every source except the textbook (see chunking.py). The
# textbook keeps its layout blocks, so its IDs and vectors do not change.
CHUNK_TARGET_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
CHUNK_MIN_CHARS = 50

# retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
USE_QUERY_PREFIX = True
RETRIEVE_KINDS = ("prose", "caption")
RETRIEVE_DOMAINS = ("clinical", "operations")   # sources.yaml "domain"
TOP_K = 5
REVIEW_POOL = 25

# generation
LLM_MODEL = "openai/gpt-oss-120b"   # Groq model id, read by generate.py
# "page":     cite as (p. 384). The original prompt; fine while the index
#             holds one paginated source.
# "numbered": cite excerpts as [2]. Needed once several sources are indexed,
#             since a page number alone no longer says which document.
CITATION_STYLE = "page"


def index_id():
    """Short ID of the current index build (written by ingest.py), or None."""
    try:
        return json.loads(MANIFEST_PATH.read_text()).get("index_id")
    except (OSError, ValueError):
        return None


def run_config():
    return {
        "embedding_model": EMBEDDING_MODEL,
        "use_query_prefix": USE_QUERY_PREFIX,
        "retrieve_kinds": list(RETRIEVE_KINDS),
        "retrieve_domains": list(RETRIEVE_DOMAINS),
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "top_k": TOP_K,
        "llm_model": LLM_MODEL,
        "citation_style": CITATION_STYLE,
        "index_id": index_id(),
    }