# app/nodes.py
from app.models import GraphState
from app.chains import Chains
from app.graders import Graders

class Nodes:
    def __init__(self, chains: Chains, retriever, web_search_tool):
        self.chains = chains
        self.retriever = retriever
        self.web_search_tool = web_search_tool

    def route_question(self, state: GraphState) -> str:
        print("---ROUTE QUESTION---")
        question = state["question"]
        history = state.get("history", [])

        if history:
            question = " \n\n".join(history + [question])
        source = self.chains.question_router.invoke({"question": question})
        if source.datasource == "retrieve":
            print("---ROUTE QUESTION TO RETRIEVER---")
            return "retrieve"
        elif source.datasource == "conversation":
            print("--ROUTE QUESTION TO CONVERSATION")
            return "conversation"
        else:
            print("---ERROR ROUTING QUESTION---")
            return "conversation"

    def init_state(self, state: GraphState) -> GraphState:
        print("\n---INIT STATE---")
        question = state["question"]
        return {"question": question, "generation": "", "documents": [], "query_rewritten_num": 0}

    def retrieve(self, state: GraphState) -> GraphState:
        print("---VECTOR DATABASE RETRIEVE---")
        question = state["question"]
        query_rewritten_num = state["query_rewritten_num"]

        # Vectorstore retrieval
        vdb_documents = self.retriever.invoke(question)
        vdb_contents = [doc.metadata.get("content") for doc in vdb_documents]

        print("---WEB SEARCH---")
        web_search_documents = self.web_search_tool.invoke({"query": question})
        web_search_contents = [d["content"] for d in web_search_documents]

        documents = vdb_contents + web_search_contents

        return {"documents": documents, "question": question, "generation": state.get("generation", ""), "query_rewritten_num": query_rewritten_num}

    def generate(self, state: GraphState) -> GraphState:
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        query_rewritten_num = state["query_rewritten_num"]

        generation = self.chains.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation, "query_rewritten_num": query_rewritten_num,"final_output": [generation]}

    def grade_documents(self, state: GraphState) -> GraphState:
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        query_rewritten_num = state["query_rewritten_num"]

        filtered_docs = []
        for d in documents:
            score = self.chains.graders.grade_documents(question, d)
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs, "question": question, "generation": state.get("generation", ""), "query_rewritten_num": query_rewritten_num}

    def transform_query(self, state: GraphState) -> GraphState:
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        query_rewritten_num = state["query_rewritten_num"]

        better_question = self.chains.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question, "generation": state.get("generation", ""), "query_rewritten_num": query_rewritten_num + 1}

    def conversation(self, state: GraphState) -> GraphState:
        print("---CONVERSATION---")
        question = state["question"]

        conversation = self.chains.conversation_chain.invoke({"question": question})
        return { "question": question, "generation": conversation, "documents": state.get("documents", []), "query_rewritten_num": state.get("query_rewritten_num", 0) }

    def decide_to_generate(self, state: GraphState) -> str:
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        query_rewritten_num = state["query_rewritten_num"]

        if not filtered_documents:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        elif query_rewritten_num > 1:   
            print("--- DECISION: TOO MANY QUERY REWRITES, END---")
            return "end"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.chains.graders.grade_hallucinations(documents, generation)
        grade = score.binary_score

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            score = self.chains.graders.grade_answer(question, generation)
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
