from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

BOOKS = {
    "mohs_layers": RAW_DIR / "MOHS_MICROGRAPHIC_SURGERY_from_layers_to_reconstruction.pdf",
}

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MIN_CHUNK_CHARS = 100