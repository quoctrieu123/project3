"""Qdrant-backed ingestion and retrieval for user-uploaded PDF documents."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langsmith import traceable

from .config import (
    DOCUMENT_CHUNK_OVERLAP,
    DOCUMENT_CHUNK_SIZE,
    DOCUMENT_UPSERT_BATCH_SIZE,
    PATH_TO_EMBEDDING,
    QDRANT_DOCUMENT_COLLECTION,
    QDRANT_DOCUMENT_HNSW_EF,
    QDRANT_DOCUMENT_TOP_K,
    QDRANT_URL,
)
from .tracing import (
    trace_cleanup_inputs,
    trace_cleanup_outputs,
    trace_document_search_inputs,
    trace_ingestion_inputs,
    trace_ingestion_outputs,
    trace_search_outputs,
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_client: QdrantClient | None = None
_embedder: SentenceTransformer | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, timeout=120)
    return _client


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(PATH_TO_EMBEDDING)
    return _embedder


def _collection_vector_size(collection_info: Any) -> int:
    vectors_config = collection_info.config.params.vectors
    if isinstance(vectors_config, dict):
        if len(vectors_config) != 1:
            raise ValueError(
                "uploaded_documents must contain exactly one dense vector"
            )
        vectors_config = next(iter(vectors_config.values()))
    return int(vectors_config.size)


def ensure_document_collection() -> None:
    """Create the document collection and filter indexes when missing."""
    client = get_qdrant_client()
    vector_size = get_embedder().get_sentence_embedding_dimension()

    if not client.collection_exists(QDRANT_DOCUMENT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_DOCUMENT_COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            on_disk_payload=True,
        )

    info = client.get_collection(QDRANT_DOCUMENT_COLLECTION)
    actual_size = _collection_vector_size(info)
    if actual_size != vector_size:
        raise ValueError(
            "Embedding dimension does not match uploaded_documents: "
            f"model={vector_size}, collection={actual_size}"
        )

    payload_schema = info.payload_schema or {}
    for field_name in ("session_id", "document_id", "file_name"):
        if field_name not in payload_schema:
            client.create_payload_index(
                collection_name=QDRANT_DOCUMENT_COLLECTION,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )


def _read_uploaded_file(uploaded_file: Any) -> tuple[str, bytes]:
    """Read a path or file-like upload without leaving its cursor changed."""
    if isinstance(uploaded_file, (str, Path)):
        path = Path(uploaded_file)
        return path.name, path.read_bytes()

    file_name = Path(getattr(uploaded_file, "name", "uploaded.pdf")).name
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        original_position = None
        if hasattr(uploaded_file, "tell"):
            try:
                original_position = uploaded_file.tell()
            except Exception:
                pass
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        data = uploaded_file.read()
        if original_position is not None and hasattr(uploaded_file, "seek"):
            uploaded_file.seek(original_position)
    else:
        raise TypeError(f"Unsupported uploaded file type: {type(uploaded_file)!r}")

    if not isinstance(data, bytes):
        raise TypeError(f"Uploaded file {file_name!r} did not provide bytes")
    return file_name, data


def chunk_text(
    text: str,
    *,
    chunk_size: int = DOCUMENT_CHUNK_SIZE,
    overlap: int = DOCUMENT_CHUNK_OVERLAP,
) -> list[str]:
    """Split normalized text into overlapping whitespace-token chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def extract_pdf_chunks(uploaded_file: Any) -> tuple[str, str, list[dict[str, Any]]]:
    """Extract page-aware chunks and return file name, document ID, and chunks."""
    file_name, file_bytes = _read_uploaded_file(uploaded_file)
    document_id = hashlib.sha256(file_bytes).hexdigest()

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Cannot read PDF {file_name!r}: {exc}") from exc

    chunks: list[dict[str, Any]] = []
    global_chunk_index = 0
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = " ".join((page.extract_text() or "").split())
        for page_chunk_index, text in enumerate(chunk_text(page_text)):
            chunks.append(
                {
                    "text": text,
                    "page": page_number,
                    "page_chunk_index": page_chunk_index,
                    "chunk_index": global_chunk_index,
                }
            )
            global_chunk_index += 1

    if not chunks:
        raise ValueError(
            f"No extractable text found in {file_name!r}; the PDF may require OCR"
        )
    return file_name, document_id, chunks


def _document_filter(session_id: str, document_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            ),
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            ),
        ]
    )


@traceable(
    name="qdrant-document-ingestion",
    run_type="tool",
    tags=["qdrant", "documents", "ingestion"],
    process_inputs=trace_ingestion_inputs,
    process_outputs=trace_ingestion_outputs,
)
def ingest_uploaded_documents(
    uploaded_files: Sequence[Any],
    session_id: str,
) -> list[str]:
    """Embed new PDFs once per session and upsert their chunks into Qdrant."""
    if not uploaded_files:
        raise ValueError("No uploaded files were provided")
    if not session_id:
        raise ValueError("session_id is required for document ingestion")

    ensure_document_collection()
    client = get_qdrant_client()
    embedder = get_embedder()
    document_ids: list[str] = []

    for uploaded_file in uploaded_files:
        file_name, document_id, chunks = extract_pdf_chunks(uploaded_file)
        document_ids.append(document_id)

        existing = client.count(
            collection_name=QDRANT_DOCUMENT_COLLECTION,
            count_filter=_document_filter(session_id, document_id),
            exact=True,
        ).count
        if existing:
            logger.info(
                "Skipping already ingested document %s for session %s",
                file_name,
                session_id,
            )
            continue

        for start in range(0, len(chunks), DOCUMENT_UPSERT_BATCH_SIZE):
            batch = chunks[start : start + DOCUMENT_UPSERT_BATCH_SIZE]
            vectors = embedder.encode(
                [chunk["text"] for chunk in batch],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            points = []
            for chunk, vector in zip(batch, vectors):
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        ":".join(
                            [
                                session_id,
                                document_id,
                                str(chunk["page"]),
                                str(chunk["page_chunk_index"]),
                            ]
                        ),
                    )
                )
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "text": chunk["text"],
                            "file_name": file_name,
                            "page": chunk["page"],
                            "page_chunk_index": chunk["page_chunk_index"],
                            "chunk_index": chunk["chunk_index"],
                            "document_id": document_id,
                            "session_id": session_id,
                        },
                    )
                )

            client.upsert(
                collection_name=QDRANT_DOCUMENT_COLLECTION,
                points=points,
                wait=True,
            )

    return document_ids


@traceable(
    name="qdrant-document-search",
    run_type="retriever",
    tags=["qdrant", "documents", "hnsw"],
    process_inputs=trace_document_search_inputs,
    process_outputs=trace_search_outputs,
)
def search_uploaded_documents(
    query: str,
    session_id: str,
    document_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Search only documents belonging to the current session and request."""
    if not query.strip():
        raise ValueError("Document query cannot be empty")
    if not session_id or not document_ids:
        raise ValueError("session_id and document_ids are required for document search")

    ensure_document_collection()
    query_vector = get_embedder().encode(
        query.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            ),
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=list(document_ids)),
            ),
        ]
    )
    points = get_qdrant_client().query_points(
        collection_name=QDRANT_DOCUMENT_COLLECTION,
        query=query_vector.tolist(),
        query_filter=query_filter,
        limit=QDRANT_DOCUMENT_TOP_K,
        search_params=models.SearchParams(
            hnsw_ef=QDRANT_DOCUMENT_HNSW_EF,
            exact=False,
        ),
        with_payload=True,
        with_vectors=False,
    ).points

    return [
        {
            "id": point.id,
            "score": float(point.score),
            "payload": dict(point.payload or {}),
        }
        for point in points
    ]


def format_document_context(results: Sequence[dict[str, Any]]) -> str:
    """Produce grounded text context with file and page citations."""
    sections: list[str] = []
    for rank, result in enumerate(results, start=1):
        payload = result.get("payload") or {}
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        sections.append(
            "\n".join(
                [
                    f"[Tài liệu liên quan #{rank}]",
                    f"File: {payload.get('file_name', 'unknown')}",
                    f"Trang: {payload.get('page', 'unknown')}",
                    f"Điểm tương đồng: {float(result['score']):.6f}",
                    f"Nội dung: {text}",
                ]
            )
        )

    if not sections:
        raise ValueError("No relevant document chunks were found for this session")
    return "\n\n".join(sections)


def retrieve_document_context(
    query: str,
    uploaded_files: Sequence[Any],
    session_id: str,
) -> str:
    """Ingest unseen PDFs, search them, and return LLM-ready context."""
    document_ids = ingest_uploaded_documents(uploaded_files, session_id)
    results = search_uploaded_documents(query, session_id, document_ids)
    return format_document_context(results)


@traceable(
    name="qdrant-session-cleanup",
    run_type="tool",
    tags=["qdrant", "cleanup"],
    process_inputs=trace_cleanup_inputs,
    process_outputs=trace_cleanup_outputs,
)
def cleanup_session_documents(session_id: str) -> int:
    """Delete every uploaded-document point owned by a conversation session."""
    if not session_id:
        return 0

    client = get_qdrant_client()
    if not client.collection_exists(QDRANT_DOCUMENT_COLLECTION):
        return 0

    session_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            )
        ]
    )
    point_count = client.count(
        collection_name=QDRANT_DOCUMENT_COLLECTION,
        count_filter=session_filter,
        exact=True,
    ).count
    if point_count:
        client.delete(
            collection_name=QDRANT_DOCUMENT_COLLECTION,
            points_selector=models.FilterSelector(filter=session_filter),
            wait=True,
        )
    return int(point_count)
