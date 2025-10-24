from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# model_name = "BAAI/bge-large-en-v1.5"
model_name = "sentence-transformers/LaBSE"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # o "cuda"
    encode_kwargs={"normalize_embeddings": False},
)

url = "http://localhost:6333"
collection_name = "gpt_db_LaBSE"

client = QdrantClient(
    url=url,
    prefer_grpc=False,
)

print(client)
print("#############################")

db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name=collection_name,
)

print(db)
print("##############################")

query = "Que tipo de imagenes utiliza la tesis titulada Sistema de apoyo para la detección de fibromas uterinos en imágenes de ultrasonido empleando redes neuronales convolucionales?"

docs = db.similarity_search_with_score(query=query, k=5)

for i in docs:
    doc, score = i
    print({"score":score, "content": doc.page_content, "metadata": doc.metadata})