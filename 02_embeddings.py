# Criação de embeddings com Hugging Face
%pip install sentence-transformers

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Criação de índice vetorial com FAISS
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    chunks,
    embeddings,
)

vectorstore.save_local("vectorstore")