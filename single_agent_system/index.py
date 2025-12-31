import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pypdf import PdfReader
import numpy as np
import re
from config import PATH_TO_EMBEDDING
import faiss
import json
from sentence_transformers import SentenceTransformer
from llm import generate_subquestion
import hashlib
import logging


_embedder = None
_laws_index = None
_laws = None
_upload_index_cache = {}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _ensure_nltk_punkt():
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


def get_laws_index_and_json():
    """Lazy-load the prebuilt laws FAISS index and the laws json list."""
    global _laws_index, _laws
    if _laws is None or _laws_index is None:
        # load json
        try:
            with open("laws_first_100k.json", 'r', encoding='utf-8') as f:
                _laws = json.load(f)
        except Exception as e:
            logger.error("Failed to load laws_first_100k.json: %s", e)
            _laws = []

        # load faiss index if available
        try:
            _laws_index = faiss.read_index("laws_first_100k.index")
        except Exception as e:
            logger.warning("Failed to read FAISS index laws_first_100k.index: %s", e)
            _laws_index = None
    return _laws_index, _laws


def _make_uploads_key(uploaded_files: list):
    """
    Create a stable key for a set of uploaded files. Uses file names.
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
    """Chunk text by approximate tokens (whitespace words) using sliding window.

    This uses whitespace splitting as an approximation for tokenization. For more
    accurate token counts use the model tokenizer.
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
    """Try nltk.sent_tokenize, fallback to naive '. ' splitting."""
    try:
        import nltk
        _ensure_nltk_punkt()
        sents = nltk.sent_tokenize(text)
        return sents
    except Exception:
        return [s.strip() for s in text.split('. ') if s.strip()]


def get_faiss_index_for_uploads(uploaded_files: list, max_tokens: int = 300, stride: int = 50):
    """
    Build faiss index for a set of uploaded files, with caching.
    """
    global _upload_index_cache
    key = _make_uploads_key(uploaded_files)
    if key in _upload_index_cache:
        return _upload_index_cache[key]

    sentences = []
    sentences_to_file = []
    for file in uploaded_files:
        reader = PdfReader(file)
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
    k = 5 if ntotal is None else min(5, int(ntotal))
    D, I = idx.search(qvec, k=k)
    # filter invalid ids
    valid_ids = [int(i) for i in I[0] if i is not None and i >= 0 and i < len(laws_list)]
    context = [laws_list[i] for i in valid_ids]
    return "\n".join(context)


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
    keys = keys[:5]
    _, laws_list = get_laws_index_and_json()
    context = [laws_list[key] for key in keys]
    return "\n".join(context)
        

def retrieve_laws_reciprocal_ranking(input_dict:dict) ->str:
    """Retrieve the laws in the vecto database based on list of queries"""
    queries = generate_subquestion(input_dict)
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
def chunk_with_overlap(sentences: list, overlap: int = 4)-> list:
    "Chunk a list of sentences with overlap"
    if overlap<1:
        return sentences
    sentences_with_overlap = []
    for i, sentence in enumerate(sentences):
        if i ==0 and i <len(sentences) -1:
            words_after = sentences[i+1].split(" ")[:overlap]
            sentences_with_overlap.append(sentence + " " + " ".join(words_after))
        elif i == len(sentences) -1:
            words_before = sentences[i-1].split(" ")[-overlap:]
            sentences_with_overlap.append(" ".join(words_before) + " " + sentence)
            break
        else:
            words_before = sentences[i-1].split(" ")[ -overlap:]
            words_after = sentences[i+1].split(" ")[:overlap]
            sentences_with_overlap.append(" ".join(words_before) + " " + sentence + " " + " ".join(words_after))
    return sentences_with_overlap

def write_from_pdf_to_faiss(uploaded_files: list, split_sentences: bool = True, max_tokens: int = 300, stride: int = 50):
    """Compatibility wrapper. Uses cached get_faiss_index_for_uploads under the hood.

    Kept for API compatibility with older code. Returns (index, sentences, sentences_to_file).
    """
    return get_faiss_index_for_uploads(uploaded_files, max_tokens=max_tokens, stride=stride)

def retrieve_sentences_documents(input_dict:dict) -> str:
    """Retrieve the sentences in the vecto database that relate with the query"""
    query = input_dict.get("query", "")
    uploaded_files = input_dict.get("uploaded_files", [])
    index, sentences, sentences_to_file = write_from_pdf_to_faiss(uploaded_files)

    if not sentences:
        logger.debug("retrieve_sentences_documents: no sentences available for uploaded files")
        return ""

    emb = get_embedder()
    qvec = np.asarray(emb.encode([query]), dtype=np.float32)

    try:
        ntotal = int(index.ntotal)
    except Exception:
        ntotal = None

    if ntotal == 0:
        logger.debug("retrieve_sentences_documents: FAISS index empty (ntotal=0)")
        return ""

    k = min(15, ntotal if ntotal is not None else 15)
    D, I = index.search(qvec, k=k)
    logger.debug("raw ids: %s", I)
    valid_ids = [int(i) for i in I[0] if i is not None and i >= 0 and i < len(sentences)]
    logger.debug("valid ids: %s", valid_ids)

    context = [sentences[i] for i in valid_ids]
    logger.debug("context chunks count=%d", len(context))
    return "\n".join(context)
