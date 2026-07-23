<<<<<<< HEAD:multi_agent_system/retrieve_laws_agent.py
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
=======
import os
import sys

# Add the project root to sys.path to allow importing from single_agent_system and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from multi_agent_langgraph.multi_agent import AgentState
import numpy as np
from single_agent_system.config import PATH_TO_EMBEDDING
import faiss
import json
from sentence_transformers import SentenceTransformer
import logging
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi

_embedder = None
_laws_index = None
_laws = None
_bm25 = None
_upload_index_cache = {}
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def get_embedder():
    """Lazy-load the SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(PATH_TO_EMBEDDING)
        logger.debug("Loaded embedder from %s", PATH_TO_EMBEDDING)
    return _embedder

def get_laws_index_and_json():
    """Lazy-load the prebuilt laws FAISS index and the laws json list."""
    global _laws_index, _laws
    if _laws is None or _laws_index is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        json_path = os.path.join(project_root, "dataset", "laws_first_100k.json")
        index_path = os.path.join(project_root, "dataset", "laws_first_100k_hnsw_v1.index")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                _laws = json.load(f)
        except Exception as e:
            logger.error("Failed to load laws_first_100k.json from %s: %s", json_path, e)
            _laws = []


        try:
            _laws_index = faiss.read_index(index_path)
            _laws_index.efSearch = 128 
        except Exception as e:
            logger.warning("Failed to read FAISS index laws_first_100k.index from %s: %s", index_path, e)
            _laws_index = None
    return _laws_index, _laws

def get_bm25():
    """Lazy-load the BM25 model."""
    global _bm25, _laws
    if _bm25 is None:
        _, laws_list = get_laws_index_and_json()
        if laws_list:
            # Simple tokenization by splitting on whitespace
            tokenized_corpus = [doc.split() for doc in laws_list]
            _bm25 = BM25Okapi(tokenized_corpus)
            logger.debug("Loaded BM25 model")
        else:
            logger.warning("No laws data available to build BM25 index")
    return _bm25

def retrieve_laws(state: AgentState) -> str:
    """Retrieve the laws using Hybrid Search (Dense + Sparse)"""
    last_message = state.get("messages", [])[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("retrieve_laws: last message is not a HumanMessage")
    query = last_message.content
    idx, laws_list = get_laws_index_and_json()
    if idx is None or not laws_list:
        logger.warning("retrieve_laws: laws index or data not available")
        return ""

    # FAISS Search
    emb = get_embedder()
    qvec = np.asarray(emb.encode([query]), dtype=np.float32)
    ntotal = getattr(idx, "ntotal", None)
    k = 10 if ntotal is None else min(10, int(ntotal))
    D, I = idx.search(qvec, k=k)
    valid_ids_faiss = [int(i) for i in I[0] if i is not None and i >= 0 and i < len(laws_list)]

    # BM25 Search
    bm25 = get_bm25()
    valid_ids_bm25 = []
    if bm25:
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        # Get top 10 indices
        top_n = np.argsort(scores)[::-1][:10]
        valid_ids_bm25 = [int(i) for i in top_n if i < len(laws_list)]

    # Combine results
    docs_dict = {
        "faiss": valid_ids_faiss,
        "bm25": valid_ids_bm25
    }
    
    return reciprocal_ranking(docs_dict)


def reciprocal_ranking(docs_dict: dict, k = 60) -> str:
    """Re-rank the retrieved documents using reciprocal rank fusion"""
    ranked_docs = {}
    for docs in docs_dict.values():
        for i,doc in enumerate(docs, 1):
            if doc not in ranked_docs:
                ranked_docs[doc] = 1/(i+k)
            else:
                ranked_docs[doc] += 1/(i+k)
    ranked_docs = dict(sorted(ranked_docs.items(), key=lambda item: item[1], reverse=True))
    keys = list(ranked_docs.keys())
    keys = keys[:10]
    #return keys
    _, laws_list = get_laws_index_and_json()
    context = [laws_list[key] for key in keys]
    return "\n".join(context)

def retrieve_laws_reciprocal_ranking(state: AgentState) -> str:
    """Retrieve the laws in the vecto database based on list of queries using Hybrid Search"""
    queries = state.get("generated_subqueries", [])

    if len(queries) > 3 and queries[2] == queries[3]:
        print("Các truy vấn con giống hệt nhau, sử dụng truy vấn con đầu tiên để truy xuất luật trực tiếp (không sử dụng reciprocal ranking).")
        context = retrieve_laws(state)
        return context
    
    if not queries:
        raise ValueError("retrieve_laws_reciprocal_ranking: No generated sub-queries found in state")
    idx, laws_list = get_laws_index_and_json()
    if idx is None or not laws_list:
        logger.warning("retrieve_laws_reciprocal_ranking: laws index or data not available")
        return ""

    emb = get_embedder()
    # FAISS Search for all queries
    qvecs = np.asarray(emb.encode(queries), dtype=np.float32)
    D, I = idx.search(qvecs, k=10)
    
    docs_dict = {}
    bm25 = get_bm25()
    
    for idx_q, (query, doc_ids) in enumerate(zip(queries, I)):
        # FAISS Results
        valid_ids_faiss = [int(i) for i in doc_ids if i is not None and i >= 0 and i < len(laws_list)]
        docs_dict[f"faiss_{idx_q}"] = valid_ids_faiss
        
        # BM25 Results
        if bm25:
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)
            top_n = np.argsort(scores)[::-1][:10]
            valid_ids_bm25 = [int(i) for i in top_n if i < len(laws_list)]
            docs_dict[f"bm25_{idx_q}"] = valid_ids_bm25
            
    context = reciprocal_ranking(docs_dict)
    return context

def run_retrieve_laws_agent(state: AgentState) -> str:
    """Run the retrieve_laws_agent and return the updated state with retrieved laws context."""
    laws_context = retrieve_laws_reciprocal_ranking(state)
    return laws_context

>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/retrieve_laws_agent.py
