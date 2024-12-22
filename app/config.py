# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration variables
CHROMA_COLLECTION_NAME = "vdb_summary_query"
CHROMA_PERSIST_DIRECTORY = "./vdb_summary_query"
OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-large"
CHAT_OPENAI_MODEL = "gpt-4o-mini"
CHAT_OPENAI_TEMPERATURE = 0
