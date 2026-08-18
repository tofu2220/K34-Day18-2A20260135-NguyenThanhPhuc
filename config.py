"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM provider (OpenRouter, OpenAI-compatible API) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def create_llm_client():
    """Create the shared OpenRouter client, or return None for offline mode."""
    if not OPENROUTER_API_KEY:
        return None

    from openai import OpenAI

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"X-OpenRouter-Title": "Lab 18 Production RAG"},
    )

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
