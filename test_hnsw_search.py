"""Test semantic law retrieval from a local Qdrant HNSW collection."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "legal_documents_v1"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a Vietnamese query and retrieve related laws with Qdrant HNSW.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Cau hoi can tim. Neu bo trong, script se yeu cau nhap tu ban phim.",
    )
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--hnsw-ef",
        type=int,
        default=128,
        help="So candidate HNSW xem xet. Gia tri cao hon thuong tang recall va latency.",
    )
    parser.add_argument(
        "--query-prefix",
        default="",
        help="Prefix tuy chon, vi du 'query: '. Database hien tai duoc embed khong prefix.",
    )
    return parser.parse_args()


def get_vector_size(collection_info: object) -> int:
    vectors_config = collection_info.config.params.vectors
    if isinstance(vectors_config, dict):
        if len(vectors_config) != 1:
            raise ValueError("Collection co nhieu named vectors; can chi ro vector name.")
        vectors_config = next(iter(vectors_config.values()))
    return int(vectors_config.size)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    if args.top_k <= 0 or args.hnsw_ef <= 0:
        raise ValueError("--top-k va --hnsw-ef phai lon hon 0")

    query = args.query or input("Nhập câu hỏi pháp luật: ").strip()
    if not query:
        raise ValueError("Câu hỏi không được để trống")

    client = QdrantClient(url=args.url, timeout=60)
    if not client.collection_exists(args.collection):
        available = [item.name for item in client.get_collections().collections]
        raise ValueError(
            f"Không tìm thấy collection {args.collection!r}. "
            f"Collections hiện có: {available}"
        )

    info = client.get_collection(args.collection)
    embedder = SentenceTransformer(args.model)
    model_vector_size = embedder.get_sentence_embedding_dimension()
    collection_vector_size = get_vector_size(info)
    if model_vector_size != collection_vector_size:
        raise ValueError(
            "Sai kích thước embedding: "
            f"model={model_vector_size}, collection={collection_vector_size}"
        )

    embedded_query = f"{args.query_prefix}{query}"
    query_vector = embedder.encode(
        embedded_query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()

    results = client.query_points(
        collection_name=args.collection,
        query=query_vector,
        limit=args.top_k,
        search_params=models.SearchParams(
            hnsw_ef=args.hnsw_ef,
            exact=False,
        ),
        with_payload=True,
        with_vectors=False,
    ).points

    print(f"\nCollection: {args.collection}")
    print(f"Trạng thái: {info.status}")
    print(f"Points: {info.points_count}; indexed: {info.indexed_vectors_count}")
    print(f"Query: {query}")
    print(f"HNSW ef: {args.hnsw_ef}; top-k: {args.top_k}")

    if not results:
        print("\nKhông tìm thấy kết quả.")
        return 0

    for rank, point in enumerate(results, start=1):
        payload = point.payload or {}
        text = payload.get("text", "<payload không có trường text>")
        print("\n" + "=" * 80)
        print(f"#{rank} | point_id={point.id} | score={point.score:.6f}")
        if payload.get("source") is not None:
            print(f"Nguồn: {payload['source']}")
        if payload.get("legacy_id") is not None:
            print(f"Legacy ID: {payload['legacy_id']}")
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
