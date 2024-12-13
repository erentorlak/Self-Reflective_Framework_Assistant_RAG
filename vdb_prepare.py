# Description: This script creates a vector database from a list of URLs.
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup as Soup
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pprint import pprint
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.schema import Document
from typing import List

urls = [
    "https://langchain-ai.github.io/langgraph/how-tos/",
]
#%%
# Initialize Document Loader
loader = RecursiveUrlLoader(
    url=urls[0], max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
)
#%%
docs = loader.load()
#%%
docs[32].page_content
#%%

class CreateSummary(BaseModel):
    """ Summary and possible queries for a document. """

    summary: str = Field(
        description="Summary of the document content in 5-7 sentences."
    )
    possible_queries: List[str] = Field(
        description="List of possible queries that users might ask about this topic."
    )

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
structured_llm_router = llm.with_structured_output(CreateSummary)

# Prompt
system = """You are an expert at technical documentation.
Analyze the given documentation and provide:
1. A concise summary (5-7 sentences)
2. List of 5-7 possible queries that users might ask about this topic.

Use the document to generate the summary and possible queries.
Query is a question or a statement that a user might ask about the topic.
Query examples should be relevant to the document content.
"""
summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Document: \n\n {document} \n\n "),
    ]
)
# Router chain
question_router = summary_prompt | structured_llm_router

#%%

result = question_router.invoke({"document": docs[32].page_content})
#%%
result.possible_queries
#%%
print(docs[32].metadata.get("source"))
#%%
# Initialize empty list to store summaries
new_docs = []

# Process each document
for doc in docs:

    # Generate summary for current document
    result = question_router.invoke({"document": doc.page_content})
    
    queries = ' '.join(result.possible_queries) # Convert list to string

    new_page_content = doc.metadata.get("title") + doc.metadata.get("description") + result.summary + queries

    new_metadata = {
        "source": doc.metadata.get("source"),
        "title": doc.metadata.get("title"),
        "description": doc.metadata.get("description"),
        "summary": result.summary,
        "possible_queries": queries,
        "content": doc.page_content,
    }

    document = Document(page_content=new_page_content, metadata=new_metadata)
    
    # Append to new list
    new_docs.append(document)

# save the new documents
new_docs[0].metadata.get("possible_queries")
#%%
vectorstore = Chroma.from_documents(
    documents=new_docs,
    collection_name="summary_vdb_col_name",
    embedding=OpenAIEmbeddings(),
    persist_directory="./summary_vdb_v2"
)
#%%
vectorstore = Chroma(persist_directory="./summary_vdb_v2", embedding_function=OpenAIEmbeddings())