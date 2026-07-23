"""Retrieve legal context from Qdrant and fuse sub-query rankings with RRF."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langsmith import traceable

from .config import (
    PATH_TO_EMBEDDING,
    QDRANT_HNSW_EF,
    QDRANT_LEGAL_COLLECTION,
    QDRANT_TOP_K_PER_QUERY,
    QDRANT_URL,
    RRF_K,
    RRF_TOP_K,
)
from .tracing import (
    trace_legal_search_inputs,
    trace_legal_search_outputs,
    trace_rrf_inputs,
    trace_search_outputs,
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_embedder: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None


def get_embedder() -> SentenceTransformer:
    """Lazy-load the same embedding model used to create the legal vectors."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(PATH_TO_EMBEDDING)
        logger.info("Loaded embedding model %s", PATH_TO_EMBEDDING)
    return _embedder


def get_qdrant_client() -> QdrantClient:
    """Create one reusable client for the local Qdrant service."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=60)
    return _qdrant_client


def _collection_vector_size(collection_info: Any) -> int:
    vectors_config = collection_info.config.params.vectors
    if isinstance(vectors_config, dict):
        if len(vectors_config) != 1:
            raise ValueError(
                "The legal collection must contain exactly one dense vector."
            )
        vectors_config = next(iter(vectors_config.values()))
    return int(vectors_config.size)


def validate_qdrant_collection(
    client: QdrantClient,
    embedder: SentenceTransformer,
) -> None:
    """Fail early when the configured collection or vector size is incorrect."""
    if not client.collection_exists(QDRANT_LEGAL_COLLECTION):
        available = [item.name for item in client.get_collections().collections]
        raise ValueError(
            f"Qdrant collection {QDRANT_LEGAL_COLLECTION!r} does not exist. "
            f"Available collections: {available}"
        )

    info = client.get_collection(QDRANT_LEGAL_COLLECTION)
    expected_size = embedder.get_sentence_embedding_dimension()
    actual_size = _collection_vector_size(info)
    if expected_size != actual_size:
        raise ValueError(
            "Embedding dimension does not match the Qdrant collection: "
            f"model={expected_size}, collection={actual_size}"
        )


@traceable(
    name="qdrant-legal-search",
    run_type="retriever",
    tags=["qdrant", "legal", "hnsw"],
    process_inputs=trace_legal_search_inputs,
    process_outputs=trace_legal_search_outputs,
)
def retrieve_laws_for_queries(
    queries: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Run one HNSW search per sub-query and retain point payloads for RRF."""
    clean_queries = [query.strip() for query in queries if query and query.strip()]
    if not clean_queries:
        raise ValueError("retrieve_laws_for_queries: queries list is empty")

    client = get_qdrant_client()
    embedder = get_embedder()

    try:
        validate_qdrant_collection(client, embedder)
        query_vectors = embedder.encode(
            clean_queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        results_by_query: dict[str, list[dict[str, Any]]] = {}
        for query, query_vector in zip(clean_queries, query_vectors):
            points = client.query_points(
                collection_name=QDRANT_LEGAL_COLLECTION,
                query=query_vector.tolist(),
                limit=QDRANT_TOP_K_PER_QUERY,
                search_params=models.SearchParams(
                    hnsw_ef=QDRANT_HNSW_EF,
                    exact=False,
                ),
                with_payload=True,
                with_vectors=False,
            ).points

            results_by_query[query] = [
                {
                    "id": point.id,
                    "score": float(point.score),
                    "payload": dict(point.payload or {}),
                }
                for point in points
            ]

        return results_by_query
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Unable to retrieve laws from Qdrant at {QDRANT_URL}: {exc}"
        ) from exc


@traceable(
    name="reciprocal-rank-fusion",
    run_type="chain",
    tags=["retrieval", "rrf"],
    process_inputs=trace_rrf_inputs,
    process_outputs=trace_search_outputs,
)
def reciprocal_rank_fusion(
    results_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    rrf_k: int = RRF_K,
    top_k: int = RRF_TOP_K,
) -> list[dict[str, Any]]:
    """Fuse sub-query rankings by stable Qdrant point ID."""
    if rrf_k < 0 or top_k <= 0:
        raise ValueError("rrf_k must be non-negative and top_k must be positive")

    fused: dict[Any, dict[str, Any]] = {}
    for results in results_by_query.values():
        for rank, result in enumerate(results, start=1):
            point_id = result["id"]
            if point_id not in fused:
                fused[point_id] = {
                    "id": point_id,
                    "rrf_score": 0.0,
                    "best_vector_score": float(result["score"]),
                    "payload": dict(result.get("payload") or {}),
                    "matched_queries": 0,
                }

            fused_result = fused[point_id]
            fused_result["rrf_score"] += 1.0 / (rrf_k + rank)
            fused_result["best_vector_score"] = max(
                fused_result["best_vector_score"],
                float(result["score"]),
            )
            fused_result["matched_queries"] += 1

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            item["rrf_score"],
            item["best_vector_score"],
        ),
        reverse=True,
    )
    return ranked[:top_k]


def format_laws_context(ranked_laws: Sequence[Mapping[str, Any]]) -> str:
    """Convert structured Qdrant results into grounded text for the LLM."""
    sections: list[str] = []
    for rank, law in enumerate(ranked_laws, start=1):
        payload = law.get("payload") or {}
        text = str(payload.get("text") or "").strip()
        if not text:
            logger.warning("Qdrant point %s has no payload.text", law.get("id"))
            continue

        source = payload.get("source", "unknown")
        legacy_id = payload.get("legacy_id", law.get("id"))
        sections.append(
            "\n".join(
                [
                    f"[Luật liên quan #{rank}]",
                    f"Point ID: {law.get('id')}",
                    f"Legacy ID: {legacy_id}",
                    f"Nguồn: {source}",
                    f"RRF score: {float(law['rrf_score']):.6f}",
                    f"Nội dung: {text}",
                ]
            )
        )

    if not sections:
        raise ValueError(
            "Qdrant returned results, but none contains payload['text']."
        )
    return "\n\n".join(sections)


def retrieve_laws_reciprocal_ranking(state: Mapping[str, Any]) -> str:
    """Retrieve, fuse, and format laws for all generated sub-queries."""
    queries = state.get("generated_subqueries", [])
    if not queries:
        raise ValueError(
            "retrieve_laws_reciprocal_ranking: no generated sub-queries found"
        )

    results_by_query = retrieve_laws_for_queries(queries)
    ranked_laws = reciprocal_rank_fusion(results_by_query)
    return format_laws_context(ranked_laws)


def run_retrieve_laws_agent(state: Mapping[str, Any]) -> str:
    """Return Qdrant legal context for the LangGraph retrieval node."""
    return retrieve_laws_reciprocal_ranking(state)
