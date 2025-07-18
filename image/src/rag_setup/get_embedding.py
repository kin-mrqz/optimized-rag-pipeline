import os
os.environ["USER_AGENT"] = "rag-pipeline/1.0 (contact: kinnmrqz@gmail.com)"
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    return embedding
