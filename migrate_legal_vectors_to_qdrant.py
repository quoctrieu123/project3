import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_ROOT / "index"

EMBEDDINGS_PATH = INDEX_DIR / "legal_embeddings_first_100k.npy"
DOCUMENTS_PATH = INDEX_DIR / "laws_first_100k.json"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_documents_v1"

BATCH_SIZE = 256


def load_source_data():
    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    with DOCUMENTS_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings phải là ma trận 2 chiều, nhận được {embeddings.shape}"
        )

    if len(embeddings) != len(documents):
        raise ValueError(
            "Số vector và văn bản không khớp: "
            f"{len(embeddings)} vectors, {len(documents)} documents"
        )

    return embeddings, documents


def ensure_collection(
    client: QdrantClient,
    vector_size: int,
):
    if client.collection_exists(COLLECTION_NAME):
        collection = client.get_collection(COLLECTION_NAME)
        print(
            f"Collection {COLLECTION_NAME!r} đã tồn tại, "
            f"points_count={collection.points_count}"
        )
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
            on_disk=True,
        ),
        on_disk_payload=True,
    )

    print(f"Đã tạo collection {COLLECTION_NAME!r}")


def migrate():
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=120,
    )

    embeddings, documents = load_source_data()

    number_of_points, vector_size = embeddings.shape

    print(f"Vectors: {number_of_points}")
    print(f"Dimension: {vector_size}")
    print(f"Dtype: {embeddings.dtype}")

    ensure_collection(client, vector_size)

    for start in tqdm(
        range(0, number_of_points, BATCH_SIZE),
        desc="Uploading legal vectors",
    ):
        end = min(start + BATCH_SIZE, number_of_points)

        points = [
            PointStruct(
                id=index,
                vector=embeddings[index].tolist(),
                payload={
                    "text": documents[index],
                    "source": "laws_first_100k",
                    "legacy_id": index,
                },
            )
            for index in range(start, end)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

    collection = client.get_collection(COLLECTION_NAME)

    print("Migration hoàn tất")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points count: {collection.points_count}")
    print(f"Indexed vectors count: {collection.indexed_vectors_count}")


if __name__ == "__main__":
    migrate()