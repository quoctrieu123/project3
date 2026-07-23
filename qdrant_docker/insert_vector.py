from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
embedder = SentenceTransformer(EMBEDDING_MODEL)
client = QdrantClient(host="localhost", port=6333)
documents = [
    {
        "id": 1,
        "text": "Nội dung điều luật thứ nhất...",
        "law_name": "Bộ luật Lao động",
        "article": "Điều 1",
    },
    {
        "id": 2,
        "text": "Nội dung điều luật thứ hai...",
        "law_name": "Bộ luật Lao động",
        "article": "Điều 2",
    },
]


passages = [
    f"passage: {document["text"]}"
    for document in documents
]

vectors = embedder.encode(
    passages,
    normalize_embeddings=True # chuẩn hóa vector embedding về độ dài = 1
).tolist()


points = [
    PointStruct(
        id=document["id"], # id của document
        vector=vector, # vector embedding của document
        payload={
            "law_name": document["law_name"],
            "article": document["article"],
            "text": document["text"],
        }, # payload: metadata của document
    )
    for document, vector in zip(documents, vectors)
]


client.upsert(
    collection_name="legal_documents",
    points=points,
    wait=True
)

COLLECTION_NAME = "legal_documents"
query = "Người lao động được nghỉ phép bao nhiêu ngày?"

query_vector = embedder.encode(
    [f"query: {query}"],
    normalize_embeddings=True,
)[0].tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5,
    with_payload=True,
).points

for result in results:
    print("Score:", result.score)
    print("Text:", result.payload["text"])
    print("Law:", result.payload.get("law_name"))
    print()