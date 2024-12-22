# app/inference.py
from typing import Iterator
from app.workflow import Workflow
from app.chains import Chains
from app.graders import Graders
from langchain_openai import ChatOpenAI
from app.nodes import Nodes
from langchain_community.tools.tavily_search import TavilySearchResults

class InferenceEngine:
    def __init__(self, chains: Chains, retriever, web_search_tool):
        self.nodes = Nodes(chains, retriever, web_search_tool)
        self.workflow = Workflow(self.nodes)
        self.compiled_workflow = self.workflow.compile_workflow()

    def inference(self, inputs: str, history, *args, **kwargs) -> Iterator[str]:
        """
        Inference Generator to support streaming
        Args:
            inputs (str): The input query for inference
            history (list[list] | list[tuple]): The chat history
            args: additional arguments
            kwargs: additional keyword arguments
        Yields:
            str: Generated text portions
        """
        thread = {"configurable": {"thread_id": "1"}}
        inputs_dict = {"question": inputs}
        
        try:
            for output in self.compiled_workflow.stream(inputs_dict, thread):
                
                for key, value in output.items():
                    print(f"Processing node: {key}")
                    if isinstance(value, dict) and "generation" in value:
                        yield value["generation"]
                    elif key == "generation":
                        yield value
                    else:
                        continue
                            
        except Exception as e:
            print(f"Error during inference: {e}")
            yield f"An error occurred: {str(e)}"
