# app/main.py
#%%
from langchain_openai import ChatOpenAI
from app.vector_store import initialize_vector_store
from langchain_community.tools.tavily_search import TavilySearchResults
from app.chains import Chains
from app.graders import Graders
from app.inference import InferenceEngine
from app.interface import Interface

def main():
    # Initialize vector store and retriever
    vectorstore, retriever = initialize_vector_store()

    # Initialize other tools
    web_search_tool = TavilySearchResults(max_results=2)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialize graders
    graders = Graders(llm)

    # Initialize chains
    chains = Chains(llm, graders)

    # Initialize inference engine
    inference_engine = InferenceEngine(chains, retriever, web_search_tool)

    # Initialize and launch interface
    interface = Interface(inference_engine)
    interface.launch_interface()

if __name__ == "__main__":
    main()

# %%
