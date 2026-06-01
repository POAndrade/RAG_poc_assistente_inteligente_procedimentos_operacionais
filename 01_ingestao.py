# Libs
%pip install langchain
%pip install -U langchain langchain-community
%pip install faiss-cpu
%pip install pypdf
%pip install openai
%pip install langchain-text-splitters


# Ler PDFs com LangChain
from langchain_community.document_loaders import TextLoader

arquivos = [
    "/Workspace/DS II/RAG_poc_assistente_inteligente_procedimentos_operacionais/docs/compensacao.txt",
    "/Workspace/DS II/RAG_poc_assistente_inteligente_procedimentos_operacionais/docs/cobranca.txt",
    "/Workspace/DS II/RAG_poc_assistente_inteligente_procedimentos_operacionais/docs/convenios.txt"
]

docs = []

for arquivo in arquivos:
    loader = TextLoader(arquivo, encoding='latin-1')
    docs.extend(loader.load())


# Quebrar em chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)


# Criação de embeddings com Hugging Face
%pip install sentence-transformers

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Criação de índice vetorial com FAISS
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    chunks,
    embeddings,
)

vectorstore.save_local("vectorstore")


# Busca semântica
docs = vectorstore.similarity_search(
    "Como funciona o proceduimento de cobrança?",
    k=1
)

for d in docs:
    print(d.page_content)