# app/workflow.py
from langgraph.graph import END, StateGraph, START
from app.models import GraphState
from app.nodes import Nodes

class Workflow:
    def __init__(self, nodes: Nodes):
        self.nodes = nodes
        self.workflow = StateGraph(GraphState)
        self.setup_workflow()

    def setup_workflow(self):
        # Define the nodes
        self.workflow.add_node("conversation", self.nodes.conversation)  # fallback conversation
        self.workflow.add_node("init_state", self.nodes.init_state)      # init
        self.workflow.add_node("retrieve", self.nodes.retrieve)          # retrieve
        self.workflow.add_node("grade_documents", self.nodes.grade_documents)  # grade documents
        self.workflow.add_node("generate", self.nodes.generate)          # generate
        self.workflow.add_node("transform_query", self.nodes.transform_query)  # transform_query

        # Build graph
        self.workflow.add_conditional_edges(
            START,
            self.nodes.route_question,
            {
                "retrieve": "init_state",
                "conversation": "conversation",
            },
        )
        self.workflow.add_edge("conversation", END)
        
        self.workflow.add_edge("init_state", "retrieve") 
        self.workflow.add_edge("retrieve", "grade_documents")
        
        self.workflow.add_conditional_edges("grade_documents",
            self.nodes.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
                "end": END,
            },
        )
        self.workflow.add_edge("transform_query", "retrieve")
        self.workflow.add_conditional_edges("generate",
            self.nodes.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
    
    def compile_workflow(self,memory_saver):
        #return self.workflow.compile(memory_saver)
        return self.workflow.compile()
