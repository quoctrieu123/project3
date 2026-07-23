"""Shared configuration for retrieval agents."""

import os


# Keep this alias for the existing PDF retrieval agent.
PATH_TO_EMBEDDING = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)

# Gemma 4 31B instruction-tuned model served by the Gemini API.
GOOGLE_GENERATIVE_MODEL = os.getenv(
    "GOOGLE_GENERATIVE_MODEL",
    "gemma-4-31b-it",
)
GOOGLE_GENERATIVE_TEMPERATURE = float(
    os.getenv("GOOGLE_GENERATIVE_TEMPERATURE", "0.2")
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_LEGAL_COLLECTION = os.getenv(
    "QDRANT_LEGAL_COLLECTION",
    "legal_documents_v1",
)
QDRANT_DOCUMENT_COLLECTION = os.getenv(
    "QDRANT_DOCUMENT_COLLECTION",
    "uploaded_documents",
)

QDRANT_TOP_K_PER_QUERY = int(os.getenv("QDRANT_TOP_K_PER_QUERY", "5"))
QDRANT_HNSW_EF = int(os.getenv("QDRANT_HNSW_EF", "128"))

RRF_K = int(os.getenv("RRF_K", "60"))
RRF_TOP_K = int(os.getenv("RRF_TOP_K", "10"))

QDRANT_DOCUMENT_TOP_K = int(os.getenv("QDRANT_DOCUMENT_TOP_K", "10"))
QDRANT_DOCUMENT_HNSW_EF = int(os.getenv("QDRANT_DOCUMENT_HNSW_EF", "128"))
DOCUMENT_CHUNK_SIZE = int(os.getenv("DOCUMENT_CHUNK_SIZE", "300"))
DOCUMENT_CHUNK_OVERLAP = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "50"))
DOCUMENT_UPSERT_BATCH_SIZE = int(os.getenv("DOCUMENT_UPSERT_BATCH_SIZE", "128"))
