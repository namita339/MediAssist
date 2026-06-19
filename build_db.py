from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

print("Loading PDF...")
loader = PyPDFLoader("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
documents = loader.load()
print(f"PDF loaded! Total pages: {len(documents)}")

print("Splitting into chunks...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print(f"Total chunks: {len(docs)}")

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS database... (may take a few minutes)")
os.makedirs("vectorstore", exist_ok=True)
db = FAISS.from_documents(docs, embedding_model)
db.save_local("vectorstore/db_faiss")
print("\n✅ FAISS DB built and saved successfully!")
print("Now run: python app.py")
