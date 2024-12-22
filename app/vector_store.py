# app/vector_store.py
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    OPENAI_EMBEDDINGS_MODEL
)

def initialize_vector_store():
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL)
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return vectorstore, retriever
