import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Root paths
# settings.py is in GISRAG/src/config/
# parents[0]=config, parents[1]=src, parents[2]=GISRAG
# ─────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parents[2]
SRC_DIR         = ROOT_DIR / "src"

# Aligning with the new Docker data structure
DATA_DIR        = ROOT_DIR / "data"
DOCUMENTS_DIR   = DATA_DIR / "input"
OUTPUTS_DIR     = DATA_DIR / "output"
CACHE_DIR       = DATA_DIR / "cache"

# Offline Models directory
MODEL_CACHE_DIR = ROOT_DIR / "models"
PROMPTS_DIR     = SRC_DIR / "prompts"

# Ensure dirs exist at import time
for _d in [DOCUMENTS_DIR, OUTPUTS_DIR, CACHE_DIR, MODEL_CACHE_DIR, PROMPTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Embedding models
#
# Priority:
#   1. EMBED_MODEL_PATH in .env  (use a HF hub ID like
#      "mixedbread-ai/mxbai-embed-large-v1", NOT a bare local
#      folder name unless that folder contains a full
#      sentence-transformers layout with 1_Pooling/config.json)
#   2. Falls back to the official HF hub ID so pooling config
#      is always resolvable.
# ─────────────────────────────────────────────────────────────
class TextEmbeddingConfig:
    # Reads from .env; falls back to the HF hub ID to guarantee
    # the pooling config is present (fixes the
    # "Pooling.__init__() missing embedding_dimension" error).
    MODEL_PATH          = os.getenv(
                            "EMBED_MODEL_PATH",
                            "mixedbread-ai/mxbai-embed-large-v1"
                          )
    VECTOR_DIM          = int(os.getenv("VECTOR_DIMENSION", "1024"))
    NORMALIZE           = True
    BATCH_SIZE          = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    QUERY_INSTRUCTION   = "Represent this sentence for searching relevant passages: "
    PASSAGE_INSTRUCTION = ""

# ─────────────────────────────────────────────────────────────
# PostgreSQL + pgvector
# ─────────────────────────────────────────────────────────────
class PostgresConfig:
    HOST     = os.getenv("PG_HOST",     "localhost")
    PORT     = int(os.getenv("PG_PORT", "5432"))
    DB       = os.getenv("PG_DB",       "gisrag_db")
    USER     = os.getenv("PG_USER",     "gisrag")
    PASSWORD = os.getenv("PG_PASSWORD", "gisrag_secret")

    # pgvector settings
    VECTOR_DIMENSION    = int(os.getenv("VECTOR_DIMENSION", "1024"))
    SIMILARITY_FUNCTION = os.getenv("SIMILARITY_FUNCTION", "cosine")
    VECTOR_INDEX_NAME   = os.getenv("VECTOR_INDEX_NAME",   "idx_records_embedding")

# ─────────────────────────────────────────────────────────────
# RAG & Chunker
# ─────────────────────────────────────────────────────────────
class ChunkConfig:
    MAX_NEIGHBORS       = 3
    MIN_PARAGRAPH_CHARS = 20

class RetrievalConfig:
    TOP_K             = 5
    SCORE_THRESHOLD   = 0.30
    MAX_CONTEXT_CHARS = 4000

# ─────────────────────────────────────────────────────────────
# LLM (Offline Llama.cpp — legacy, kept for local GGUF usage)
# ─────────────────────────────────────────────────────────────
class LLMConfig:
    PROVIDER     = "local"
    MODEL_PATH   = str(MODEL_CACHE_DIR / "gemma-4-E4B" / "google_gemma-4-E4B-it-Q4_0.gguf")
    N_CTX        = 8192
    N_GPU_LAYERS = -1   # -1 offloads all possible layers to GPU
    MAX_TOKENS   = 1024
    TEMPERATURE  = 0.2
    STOP_TOKENS  = ["<channel|>", "<turn|>", "<eos>"]

# ─────────────────────────────────────────────────────────────
# Ollama  (RAG pipeline default — gemma3:4b)
# ─────────────────────────────────────────────────────────────
class OllamaConfig:
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL    = os.getenv("OLLAMA_MODEL",    "gemma3:4b")
    TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ─────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────
CLEANED_RECORDS_PATH = Path(
    os.getenv("CLEANED_OUTPUT_FOLDER", str(OUTPUTS_DIR))
) / "cleaned_records.json"

# ─────────────────────────────────────────────────────────────
# Convenience bundle
# ─────────────────────────────────────────────────────────────
class Settings:
    root                 = ROOT_DIR
    documents            = DOCUMENTS_DIR
    outputs              = OUTPUTS_DIR
    cache                = CACHE_DIR
    model_cache          = MODEL_CACHE_DIR
    cleaned_records_path = CLEANED_RECORDS_PATH

    text_embedding = TextEmbeddingConfig()
    postgres       = PostgresConfig()
    chunk          = ChunkConfig()
    retrieval      = RetrievalConfig()
    llm            = LLMConfig()
    ollama         = OllamaConfig()
    prompts_dir    = PROMPTS_DIR

settings = Settings()