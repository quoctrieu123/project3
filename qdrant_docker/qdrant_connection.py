from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams # cách tính độ giống nhau và mô tả vector được lưu trong hệ thống
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "legal_documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large" # model embedding vector

client = QdrantClient(host="localhost", port=6333) # 1 collection = 1 database trong qdrant
embedder = SentenceTransformer(EMBEDDING_MODEL)

vector_size = embedder.get_sentence_embedding_dimension() # lấy số chiều của model embedding

# nếu collection chưa tồn tại thì tạo mới collection
if not client.collection_exists(COLLECTION_NAME):
    client.recreate_collection(
        collection_name=COLLECTION_NAME, # tên collection
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE), # khai báo vector size và cách tính độ giống nhau giữa các vector (cosine similarity)
    )

