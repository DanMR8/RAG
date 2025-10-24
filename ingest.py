from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cargar PDF
loader = PyPDFLoader("data.pdf")
documents = loader.load()

# Dividir texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# Embeddings
# model_name = "BAAI/bge-large-en-v1.5"
model_name = "sentence-transformers/LaBSE"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # o "cuda"
    encode_kwargs={"normalize_embeddings": False},
)

print("Embeddings model loaded...")

# Conexión Qdrant
url = "http://localhost:6333"
collection_name = "gpt_db_LaBSE"

# Crear colección
qdrant = Qdrant.from_documents(
    documents=texts,        # <-- nombre correcto
    embedding=embeddings,   # <-- parámetro correcto
    url=url,
    prefer_grpc=False,
    collection_name=collection_name,
)

print("Qdrant index created successfully!")
