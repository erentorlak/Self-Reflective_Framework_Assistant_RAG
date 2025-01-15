#%% Imports and Setup
from datetime import time
from typing import Annotated, List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_core.documents import Document
from typing import Literal, List
import gradio as gr
import langsmith
from pprint import pprint
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
client = langsmith.Client()

#%% Vector Database Setup
vectorstore = Chroma(
    collection_name="vdb_summary_query",
    persist_directory="./vdb_summary_query",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
web_search_tool = TavilySearchResults(max_results=2)

#%% Data Models
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["retrieve", "conversation"] = Field(
        ...,
        description="Given a user question choose to route it to retrieve or conversation.",
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
from langgraph.graph.message import add_messages


def add_summary(messages: str, summary: str) -> str:        # custom reducer function for chat history
    """
    Concatenates the summary to the messages.

    Args:
        messages (str): The messages to append the summary to
        summary (str): The summary to append

    Returns:

        str: The concatenated messages
    """
    return messages + "\n" + summary




class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]    
    query_rewritten_num: int
    re_generation_num: int
    #summary: str
    summary: Annotated[str, add_summary]
    #summary: Annotated[list, add_messages]

    


#%% Prompts and Chains
# Router prompt
system = """You are an expert at routing a user question to a retrieve or conversation.
The retrieve contains documentation of Langchain and LangGraph.

LangGraph is a library for building stateful, multi-agent applications with LLMs, used to create agent and multi-agent workflows. 

Use retrieve if the user asks questions about:
- Details about LangGraph
- Usage examples and API documentation for LangGraph/Langchain
- Implementation details and source code questions
- Any agent or multi-agent workflow questions

If a user uses daily conversation, then use conversation."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Grader prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader


structured_llm_router = llm.with_structured_output(RouteQuery)
question_router = route_prompt | structured_llm_router
# Generate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("heyheyerent/erent_promptv1")

# LLM
llm = ChatOpenAI( model="gpt-4o-mini", temperature=0.5)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Hallucination Grader
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

structured_llm_grader = llm.with_structured_output(GradeHallucinations)
hallucination_grader = hallucination_prompt | structured_llm_grader

# Answer Grader
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

structured_llm_grader = llm.with_structured_output(GradeAnswer)
answer_grader = answer_prompt | structured_llm_grader

# Fallback Conversation 
system = """You are a helpful assistant that can answer questions on a wide range of topics. \n
    Your input will be a summary of the conversation so far that you can use to guide your response. \n
    Then the question will be asked to you. You can answer the question or ask for clarification. \n
    """
    
conv_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Context and Question : {question}")
    ]
)

# LLM
conv_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)

# Conversation Chain
conversation_chain = conv_prompt | conv_llm | StrOutputParser()

# Question Re-writer
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

#%% Node Functions
def conversation(state):
    """
    Acts as a fall back conversation chain. 

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---CONVERSATION---")

    summary = state.get("summary", "")  # get the summary from the state if it exists

    question = state["question"]

    # concat summary and question
    sum_and_q = summary + "\nQuestion : " + question

    
    # Conversation
    generation = conversation_chain.invoke({"question": sum_and_q})

    summary = "Question : " + question + "." + " AI Response : " + generation
    return { "question": question, "generation": generation,"query_rewritten_num": 0, "summary": summary}

def init_state(state):
    """
    Initialize the graph state
    """


    print("\n---INIT STATE---")
    question = state["question"]
    
    return {"question": question, "query_rewritten_num":0, "re_generation_num":0}
    #return {"question": question, "generation": "", "documents": [], "query_rewritten_num": 0}


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    
    print("---VECTOR DATABASE RETRIEVE---")
    question = state["question"]
    question = state["question"]
    query_rewritten_num = state["query_rewritten_num"]

    # vectorstore retrieval
    vdb_documents = retriever.invoke(question)
    vdb_contents = [doc.metadata.get("content") for doc in vdb_documents]  # this is a list of strings


    print("---WEB SEARCH---")

    # Web search
    web_search_documents = web_search_tool.invoke({"query": question})
    #web_search_contents = [d["content"] for d in web_search_documents]

    if isinstance(web_search_documents, list) and all(isinstance(d, dict) and "content" in d for d in web_search_documents):
        web_search_contents = [d["content"] for d in web_search_documents]
    else:
        print("Unexpected structure of web_search_documents:", web_search_documents)
        web_search_contents = []  # Fallback to an empty list or handle differently


    

    documents = vdb_contents + web_search_contents


    return {"documents": documents, "question": question, "query_rewritten_num":query_rewritten_num}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    query_rewritten_num = state["query_rewritten_num"]

    summary = state.get("summary", "")  # get the summary from the state if it exists

    sum_and_q = summary + "\nAnswer the Question : " + question

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": sum_and_q})

    summary = "Question : " + question + "." + " AI Response : " + generation

    return {"documents": documents, "question": question, "generation": generation, "query_rewritten_num":query_rewritten_num, "summary": summary}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    query_rewritten_num = state["query_rewritten_num"]
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue


    # maybe control for if filtered_docs is empty : bunu aşağıda kontrol ediyoruz decide_to_generate fonksiyonunda


    return {"documents": filtered_docs, "question": question, "query_rewritten_num": query_rewritten_num}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    query_rewritten_num = state["query_rewritten_num"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question, "query_rewritten_num": query_rewritten_num + 1}

#%% Edge Functions
def route_question(state):
    print("---ROUTE QUESTION---")
    question = state["question"],
    source = question_router.invoke({"question": question})
    if source.datasource == "retrieve":
        print("---ROUTE QUESTION TO RETRIEVER---")
        return "retrieve"
    elif source.datasource == "conversation":
        print("--ROUTE QUESTION TO CONVERSATION")
        return "conversation"
    else:
        print("---ERROR ROUTING QUESTION---")
        return "conversation"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]
    query_rewritten_num = state["query_rewritten_num"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    elif query_rewritten_num > 1:   
        # We have re-written the query too many times
        print("--- DECISION: TOO MANY QUERY REWRITES, END---")
        return "end"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):   
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    re_generation_num = state.get("re_generation_num", 0)
    #is_websearch = state["is_websearch"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}  
    )
    grade = score.binary_score

    if re_generation_num > 2:
        print("---DECISION: TOO MANY RE-GENERATIONS, END---")
        return "useful"

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        state["re_generation_num"] = re_generation_num + 1
        return "not supported"

#%% Graph Construction
workflow = StateGraph(GraphState)
memory = MemorySaver()

# Add nodes
workflow.add_node("conversation", conversation)  # fallback conversation

workflow.add_node("init_state", init_state)  # init
workflow.add_node("retrieve", retrieve)  # retrieve

workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Add edges
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "retrieve": "init_state",
        "conversation": "conversation",
    },
)
workflow.add_edge("conversation", END)

workflow.add_edge("init_state", "retrieve") 
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges("grade_documents",
    
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "end": END,
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges("generate",
    
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

app = workflow.compile(checkpointer=memory)

#%% Gradio Interface
def inference(inputs, history, *args, **kwargs):
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
    inputs = {"question": inputs}
    
    try:
        for output in app.stream(inputs, thread):
            # Debug output
            #print("Output received:", output)
            
            for key, value in output.items():
                #print(f"Processing node: {key}")
                # First yield the node name
                yield f"[{key}]\n"
                
                if isinstance(value, dict) and "generation" in value:                                           #direkt key==generate yapmak lazım
                    yield value["generation"]   
                elif key == "generation":
                    yield value
                else:
                    continue
    except Exception as e:
        print(f"Error during inference: {e}")
        yield f"An error occurred: {str(e)}"

# Update Gradio interface
demo = gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(height=600), 
    textbox=gr.Textbox(placeholder="Ask a question", container=False, scale=7),
    title="Self Reflective Framework Assistant",
    description="Ask me anything about LangGraph."
)
#%% Evaluation Functions
demo.launch()
#%% Evaluation Functions































#%% Evaluation Functions
class ScoreModel(BaseModel):
    """Relevance and performance score for a response"""

    score: int = Field(
        ...,
        description="Score between 0-5",
        ge=0,   # greater or equal 
        le=5    # less or equal
    )
    justification: str = Field(
        ...,
        description="Justification for the score"
    )
    
# Function to score relevance using LLM
def score_relevance(query, response):
    scoring_prompt = f"""
    You are a grader assessing the relevance of a response to a query.
    In no way response should contain mock up or hallucination.

    Response should not be a hypothetical answer. 

    If question can be answered with the code and response contains code, it is relevant but if it is not, it is not relevant.

    0: Hypothetical answer or irrelevant
    1: Slightly relevant
    2: Moderately relevant
    3: Relevant
    4: Highly relevant
    5: Contains real code example and directly answers the question

    Dont be generous, be strict.
    You should tend to give lower scores with no code responses.

    If the response doesnt need any code to answer the question make sure answer is confidently answered score accordingly.
    
    If the response is saying that like "I am not sure, not explicitly mentioned, I think, I believe" etc. give lower scores.
    Query: {query}
    Response: {response}
    Score the relevance on a scale of 0-5.
    """
    scoring_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    scoring_llm = scoring_llm.with_structured_output(ScoreModel)
    score = scoring_llm.invoke(scoring_prompt)
    return score

# Function to measure latency and relevance
def measure_performance(query):
    results = {}

    # Measure Standard LLM
    llm_base = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = """
    You are an assistant for question-answering tasks and coding tasks.
    If you don't know the answer, just say that you don't know.
    If it is asked how to code then you should code and if you are not sure about the code then you should say that you are not sure.
    If it is not asked to code then keep the answer concise.
    Question: {question}
    Answer:
    """

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} Answer it : "),
    ]
)
    llm_base_with_prompt = prompt | llm_base 
    start_time = time.time()
    response = llm_base_with_prompt.invoke(query)
    end_time = time.time()

    rel_score = score_relevance(query, response.content)
    
    relevance = rel_score.score
    justification = rel_score.justification

    results["Standard LLM"] = {
        "response": response.content,
        "latency": end_time - start_time,
        "relevance": relevance,
        "justification": justification
    }
    
    # Measure Normal RAG  
    start_time = time.time()
    docs = retriever.invoke(query)
    docs_content = "\n\n".join(doc.metadata.get("content") for doc in docs)
    response = rag_chain.invoke({"context": docs_content, "question": query})
    end_time = time.time()

    rel_score = score_relevance(query, response)
    
    relevance = rel_score.score
    justification = rel_score.justification

    results["Normal RAG"] = {
        "response": response,
        "latency": end_time - start_time,
        "relevance": relevance,
        "justification": justification
    }

    # Measure Self-RAG
    start_time = time.time()
    thread = {"configurable": {"thread_id": "1"}} 
    inputs = {"question": query}
    for output in app.stream(inputs, thread):
        for key, value in output.items():
            if key == "generate":
                response = value
    end_time = time.time()

    rel_score = score_relevance(query, response)
    
    relevance = rel_score.score
    justification = rel_score.justification

    results["Self-RAG"] = {
        "response": response.get("generation"),
        "latency": end_time - start_time,
        "relevance": relevance,
        "justification": justification
    }

    return results

#save performance results to json
import json

# Add after the existing performance measurement code
test_queries = [
    "Give me code example that uses ToolNode in langgraph",
    "What is Command in langgraph?",
    "What is the difference between Dag and LangGraph?"
    #"Langgraph adaptive rag local code example",
]

def evaluate_multiple_queries():
    all_results = {}
    summary = []
    
    for query in test_queries:
        print(f"\nEvaluating query: {query}")
        results = measure_performance(query)
        all_results[query] = results
        
        # Create summary entry for each method
        for method, data in results.items():
            summary.append({
                'query': query,
                'method': method,
                'relevance': data['relevance'],
                'latency': data['latency'],
                'justification': data['justification']
            })
    
    return all_results, summary

def display_evaluation_summary(summary):
    print("\nEvaluation Summary:")
    print("=" * 100)
    print(f"{'Query':<50} | {'Method':<15} | {'Relevance':<9} | {'Latency':<8} | {'Justification'}")
    print("-" * 100)
    
    for entry in summary:
        print(f"{entry['query'][:47]+'...':<50} | "
              f"{entry['method']:<15} | "
              f"{entry['relevance']:<9} | "
              f"{entry['latency']:.2f}s | "
              f"{entry['justification'][:50]}...")

# Run evaluation
all_results, summary = evaluate_multiple_queries()

# Display summary
display_evaluation_summary(summary)

# Save detailed results to JSON
with open("performance_results_multiple.json", "w") as f:
    json.dump(all_results, f, indent=2)

# create graphics for evaluation summary
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load detailed results
with open("performance_results_multiple.json", "r") as f:
    all_results = json.load(f)

# Create a DataFrame for plotting
data = []
for query, results in all_results.items():
    for method, result in results.items():
        data.append({
            "query": query,
            "method": method,
            "latency": result["latency"],
            "relevance": result["relevance"]
        })

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x="query", y="latency", hue="method", data=df)
plt.title("Latency Comparison")
plt.ylabel("Latency (s)")
plt.xticks(rotation=45)

plt.figure(figsize=(12, 6))
sns.barplot(x="query", y="relevance", hue="method", data=df)
plt.title("Relevance Comparison")
plt.ylabel("Relevance Score")
plt.xticks(rotation=45)

plt.show()

# create a  file for evaluation summary it will be used in the report. it should be good looking readable format.
# Create evaluation summary report
def create_evaluation_report(summary, filename="evaluation_summary.md"):
    with open(filename, "w") as f:
        f.write("# Evaluation Summary Report\n\n")
        
        # Write averages section
        f.write("## Average Metrics\n\n")
        df = pd.DataFrame(summary)
        avg_metrics = df.groupby('method').agg({
            'relevance': 'mean',
            'latency': 'mean'
        }).round(2)
        
        f.write("| Method | Avg Relevance | Avg Latency (s) |\n")
        f.write("|--------|---------------|----------------|\n")
        for method, row in avg_metrics.iterrows():
            f.write(f"| {method} | {row['relevance']} | {row['latency']} |\n")
        
        # Write detailed results section
        f.write("\n## Detailed Results\n\n")
        for entry in summary:
            f.write(f"### Query: {entry['query']}\n")
            f.write(f"- Method: {entry['method']}\n")
            f.write(f"- Relevance Score: {entry['relevance']}\n")
            f.write(f"- Latency: {entry['latency']:.2f}s\n")
            f.write(f"- Justification: {entry['justification']}\n\n")

# Generate the report
create_evaluation_report(summary)

#%% Run Application (if needed)
if __name__ == "__main__":
    demo.launch()

# %%
# draw the graph

from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
