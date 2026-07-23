<<<<<<< HEAD:multi_agent_system/extract_docs_agent.py
"""LangGraph node for Qdrant-backed uploaded-document retrieval."""

from langchain_core.messages import HumanMessage

from .document_store import retrieve_document_context
from .multi_agent import AgentState


def run_docs_agent(state: AgentState) -> str:
    """Return relevant uploaded PDF chunks for the latest user query."""
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("run_docs_agent: messages list is empty")

    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("run_docs_agent: last message is not a HumanMessage")

    uploaded_files = state.get("uploaded_files", [])
    if not uploaded_files:
        raise ValueError("run_docs_agent: no uploaded files found in state")

    session_id = state.get("session_id", "")
    if not session_id:
        raise ValueError("run_docs_agent: session_id is missing from state")

    return retrieve_document_context(
        query=last_message.content,
        uploaded_files=uploaded_files,
        session_id=session_id,
    )

=======
import os
import sys

# Add the project root to sys.path to allow importing from single_agent_system and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from langchain_core.messages import HumanMessage
from pypdf import PdfReader
import numpy as np
import re
from single_agent_system.config import PATH_TO_EMBEDDING
import faiss
from sentence_transformers import SentenceTransformer
import logging
from multi_agent_langgraph.multi_agent import AgentState

_embedder = None
_laws_index = None
_laws = None
_upload_index_cache = {}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _ensure_nltk_punkt():
    """Function to ensure nltk is available"""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            nltk.download("punkt")
    except Exception:
        logger.debug("nltk not available; falling back to simple sentence splitter")


def get_embedder():
    """Lazy-load the SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(PATH_TO_EMBEDDING)
        logger.debug("Loaded embedder from %s", PATH_TO_EMBEDDING)
    return _embedder


def _make_uploads_key(uploaded_files: list):
    """
    Create a stable key for a set of uploaded files based on their names. 
    This is used for caching purposes.
    Args:
        uploaded_files (list): List of uploaded file objects.
    Returns:
        tuple: A tuple of sorted file names representing the key.
    """
    if not uploaded_files:
        return ("__empty__",)
    names = []
    for f in uploaded_files:
        name = getattr(f, "name", None) or str(f)
        names.append(name)
    # sort to make order-insensitive
    names = tuple(sorted(names))
    return names


def chunk_text_by_tokens(text: str, max_tokens: int = 300, stride: int = 50):
    """
    Chunk text by approximate tokens (whitespace words) using sliding window.
    This uses whitespace splitting as an approximation for tokenization. For more
    accurate token counts use the model tokenizer.
    Args:
        text (str): The input text to be chunked.
        max_tokens (int): Maximum number of tokens per chunk.
        stride (int): Number of overlapping tokens between chunks.
    Returns:
        list: List of text chunks.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [" ".join(words)]
    chunks = []
    step = max_tokens - stride if max_tokens > stride else max_tokens
    for start in range(0, len(words), step):
        window = words[start:start + max_tokens]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + max_tokens >= len(words):
            break
    return chunks


def split_into_sentences(text: str):
    """
    Try nltk.sent_tokenize, fallback to naive '. ' splitting.
    Args:
        text (str): The input text to be split into sentences.
    Returns:
        list: List of sentences.
    """
    try:
        import nltk
        _ensure_nltk_punkt()
        sents = nltk.sent_tokenize(text)
        return sents
    except Exception:
        return [s.strip() for s in text.split('. ') if s.strip()]


def get_faiss_index_for_uploaded_files(uploaded_files: list, max_tokens: int = 300, stride: int = 50):
    """
    Build faiss index for a set of uploaded files, with caching.
    Args:
        uploaded_files (list): List of uploaded file objects.
        max_tokens (int): Maximum number of tokens per chunk.
        stride (int): Number of overlapping tokens between chunks.
    Returns:
        tuple: A tuple containing the faiss index, list of sentences, and list mapping sentences to files.
    """
    global _upload_index_cache
    key = _make_uploads_key(uploaded_files)
    if key in _upload_index_cache:
        return _upload_index_cache[key]

    sentences = []
    sentences_to_file = []
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
        except Exception as e:
            raise ValueError(f"get_faiss_index_for_uploaded_files: Failed to read PDF file {file}: {e}")
        file_name = getattr(file, "name", None) or str(file)
        for page in reader.pages:
            text = page.extract_text()
            text = re.sub(r'\n+', ' ', text) if text else text
            text = re.sub(r'  +', ' ', text)
            if not text:
                continue


            page_chunks = chunk_text_by_tokens(text, max_tokens=max_tokens, stride=stride)
            if not page_chunks:
                sents = split_into_sentences(text)
                page_chunks = chunk_text_by_tokens(" ".join(sents), max_tokens=max_tokens, stride=stride)

            sentences.extend(page_chunks)
            sentences_to_file.extend([file_name] * len(page_chunks))

    if not sentences:
        emb = get_embedder()
        dim = emb.get_sentence_embedding_dimension()
        empty_index = faiss.IndexFlatL2(dim)
        _upload_index_cache[key] = (empty_index, [], [])
        return empty_index, [], []

    emb = get_embedder()
    embeddings = emb.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)

    _upload_index_cache[key] = (idx, sentences, sentences_to_file)
    return idx, sentences, sentences_to_file

def run_docs_agent(state: AgentState) -> str:
    """
    Run the agent that retrieves the sentences in the vecto database that relate with the query
    Args:
        state (AgentState): The current state of the agent containing messages and uploaded files.
    
    Returns:
        str: Retrieved document context related to the query.
    """
    messages = state.get("messages", [])
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("run_docs_agent: last message is not a HumanMessage")
    query = last_message.content.lower()
    uploaded_files = state.get("uploaded_files", [])
    
    if not uploaded_files:
        raise ValueError("run_docs_agent: No uploaded files found in state")
    
    index, sentences, sentences_to_file = get_faiss_index_for_uploaded_files(uploaded_files)

    if not sentences:
        raise ValueError("run_docs_agent: No sentences extracted from uploaded files")

    emb = get_embedder()
    qvec = np.asarray(emb.encode([query]), dtype=np.float32)

    try:
        ntotal = int(index.ntotal)
    except Exception:
        ntotal = None

    if ntotal == 0:
        raise ValueError("run_docs_agent: FAISS index empty (ntotal=0)")

    k = min(10, ntotal if ntotal is not None else 10)
    D, I = index.search(qvec, k=k)
    logger.debug("raw ids: %s", I)
    valid_ids = [int(i) for i in I[0] if i is not None and i >= 0 and i < len(sentences)]
    logger.debug("valid ids: %s", valid_ids)

    context = [sentences[i] for i in valid_ids]
    logger.debug("context chunks count=%d", len(context))
    return '\n'.join(context)
>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/extract_docs_agent.py
