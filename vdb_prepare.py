#%% 
# Description: This script creates a vector database from a list of URLs.
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from bs4 import BeautifulSoup as Soup
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.schema import Document
from typing import List
import json

#%% 
# Define URLs
urls = [
    "https://langchain-ai.github.io/langgraph/how-tos/",
]

#%% 
# Initialize Document Loader
loader = RecursiveUrlLoader(
    url=urls[0], max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
)

#%% 
# Load documents
docs = loader.load()

#%% 
# Inspect a sample document
docs[32].page_content

#%% 
# Define LLM schema for structured outputs
class CreateSummary(BaseModel):
    """Summary and possible queries for a document."""
    summary: str = Field(description="Summary of the document content in 5-7 sentences.")
    possible_queries: List[str] = Field(
        description="List of possible queries that users might ask about this topic."
    )

# Initialize ChatOpenAI with structured output
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_router = llm.with_structured_output(CreateSummary)

#%% 
# Create Prompt Template
system = """You are an expert at technical documentation.
Analyze the given documentation and provide:
1. A concise summary (5-7 sentences)
2. List of 5-7 possible queries that users might ask about this topic.
Use the document to generate the summary and possible queries.
"""
summary_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Document: \n\n {document} \n\n ")]
)
question_router = summary_prompt | structured_llm_router

#%% 
# Generate summary for a single document
result = question_router.invoke({"document": docs[32].page_content})
result.summary

#%% 
# Inspect document source metadata
print(docs[32].metadata.get("source"))

#%% 
# Process all documents and store enhanced versions
new_docs = []
for doc in docs:
    result = question_router.invoke({"document": doc.page_content})
    queries = ' '.join(result.possible_queries)  # Convert queries list to string
    new_page_content = (
        doc.metadata.get("title", "") + 
        doc.metadata.get("description", "") + 
        result.summary + 
        queries
    )
    new_metadata = {
        "source": doc.metadata.get("source"),
        "title": doc.metadata.get("title"),
        "description": doc.metadata.get("description"),
        "summary": result.summary,
        "possible_queries": queries,
        "content": doc.page_content,
    }
    document = Document(page_content=new_page_content, metadata=new_metadata)
    new_docs.append(document)

#%% 
# Save new documents to JSON (optional)
# with open("new_docs.json", "w") as f:
#     json.dump([doc.dict() for doc in new_docs], f, indent=4)

# Reload documents from JSON (optional)
# with open("new_docs.json", "r") as f:
#     loaded_docs = json.load(f)
# new_docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in loaded_docs]

#%% 
# Create Vector Database
vectorstore = Chroma.from_documents(
    documents=new_docs,
    collection_name="vdb_summary_query",
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    persist_directory="./vdb_summary_query"
)

#%% 
# Reload Vector Database
vectorstore = Chroma(
    collection_name="vdb_summary_query",
    persist_directory="./vdb_summary_query",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
)
