import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from multi_agent import AgentState
import numpy as np
from config import PATH_TO_EMBEDDING
import faiss
import json
from sentence_transformers import SentenceTransformer
import logging

_embedder = None
_laws_index = None
_laws = None
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
        
        json_path = os.path.join(project_root, "laws_first_100k.json")
        index_path = os.path.join(project_root, "laws_first_100k_ivfpq_v2.index")

        # load json
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                _laws = json.load(f)
        except Exception as e:
            logger.error("Failed to load laws_first_100k.json from %s: %s", json_path, e)
            _laws = []

        # load faiss index if available
        try:
            _laws_index = faiss.read_index(index_path)
            _laws_index.nprobe = 1
        except Exception as e:
            logger.warning("Failed to read FAISS index laws_first_100k.index from %s: %s", index_path, e)
            _laws_index = None
    return _laws_index, _laws

def retrieve_laws(input_dict:dict) -> str:
    """Retrieve the laws in the vecto database that relate with the query"""
    query = input_dict.get("query", "")
    idx, laws_list = get_laws_index_and_json()
    if idx is None or not laws_list:
        logger.warning("retrieve_laws: laws index or data not available")
        return ""

    emb = get_embedder()
    qvec = np.asarray(emb.encode([query]), dtype=np.float32)
    ntotal = getattr(idx, "ntotal", None)
    k = 10 if ntotal is None else min(10, int(ntotal))
    D, I = idx.search(qvec, k=k)
    # filter invalid ids
    valid_ids = [int(i) for i in I[0] if i is not None and i >= 0 and i < len(laws_list)]
    return valid_ids
    '''
    context = [laws_list[i] for i in valid_ids]
    return "\n".join(context)
    '''

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
    #sort the five docs with the highest score
    keys = list(ranked_docs.keys())
    keys = keys[:10]
    return keys
    '''
    _, laws_list = get_laws_index_and_json()
    context = [laws_list[key] for key in keys]
    return "\n".join(context)
    '''
        

def retrieve_laws_reciprocal_ranking(state: AgentState) -> str:
    """Retrieve the laws in the vecto database based on list of queries"""
    queries = state.get("generated_subqueries", [])
    if not queries:
        raise ValueError("retrieve_laws_reciprocal_ranking: No generated sub-queries found in state")
    idx, laws_list = get_laws_index_and_json()
    if idx is None or not laws_list:
        logger.warning("retrieve_laws_reciprocal_ranking: laws index or data not available")
        return ""

    emb = get_embedder()
    qvecs = np.asarray(emb.encode(queries), dtype=np.float32)
    D, I = idx.search(qvecs, k=5)
    docs_dict = {}
    for query, doc_ids in zip(queries, I):
        valid_ids = [int(i) for i in doc_ids if i is not None and i >= 0 and i < len(laws_list)]
        docs_dict[query] = valid_ids
    context = reciprocal_ranking(docs_dict)
    return context

def run_retrieve_laws_agent(state: AgentState) -> str:
    """Run the retrieve_laws_agent and return the updated state with retrieved laws context."""
    laws_context = retrieve_laws_reciprocal_ranking(state)
    return laws_context

