from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_DIR = ROOT / "data" / "eval"
FIGURES_DIR = PROCESSED_DIR / "figures"

BOOKS = {
    "mohs_layers": RAW_DIR / "MOHS_MICROGRAPHIC_SURGERY_from_layers_to_reconstruction.pdf",
}

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
VECTORS_PATH = PROCESSED_DIR / "vectors.npy"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
REVIEW_SHEET_PATH = EVAL_DIR / "review_sheet.md"
RESULTS_PATH = EVAL_DIR / "results.jsonl"

# ingestion
MIN_CHUNK_CHARS = 100
MIN_FIGURE_PIXELS = 150

# retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
USE_QUERY_PREFIX = True
RETRIEVE_KINDS = ("prose", "caption")
TOP_K = 5
REVIEW_POOL = 25

# generation
LLM_MODEL = "qwen2.5:7b"


def run_config():
    return {
        "embedding_model": EMBEDDING_MODEL,
        "use_query_prefix": USE_QUERY_PREFIX,
        "retrieve_kinds": list(RETRIEVE_KINDS),
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "top_k": TOP_K,
    }