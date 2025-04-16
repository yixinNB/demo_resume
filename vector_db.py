from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

documents = [
    """
    """,
]

# 文本预处理
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=50
# )
# docs = text_splitter.create_documents(documents)
documents_ = [
    Document(
        page_content=text,
        metadata={}  # 可以添加元数据，例如 {"source": "web"}
    )
    for text in documents
]

# 生成向量库
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_db = FAISS.from_documents(documents_, embeddings)
# vector_db.save_local("resume_faiss")
