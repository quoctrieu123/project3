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

