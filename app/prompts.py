# app/prompts.py
from langchain_core.prompts import ChatPromptTemplate

# Routing Prompt
ROUTE_SYSTEM_PROMPT = """You are an expert at routing a user question to a retrieve or conversation.
The retrieve contains documentation of Langchain and LangGraph.

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence.Additionally, LangGraph includes built-in persistence, enabling advanced human-in-the-loop and memory features.Key Features
Cycles and Branching: Implement loops and conditionals in your apps.
Persistence: Automatically save state after each step in the graph. Pause and resume the graph execution at any point to support error recovery, human-in-the-loop workflows, time travel and more.
Human-in-the-Loop: Interrupt graph execution to approve or edit next action planned by the agent.
Streaming Support: Stream outputs as they are produced by each node (including token streaming).
Use the retrieve for questions on these topics only. 

If a user uses informal conversation, then use conversation."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTE_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# Grader Prompts
GRADE_DOCUMENTS_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

GRADE_HALLUCINATIONS_SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

GRADE_ANSWER_SYSTEM_PROMPT = """You are a grader assessing whether an answer addresses / resolves a question. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

grade_documents_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADE_DOCUMENTS_SYSTEM_PROMPT),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADE_HALLUCINATIONS_SYSTEM_PROMPT),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADE_ANSWER_SYSTEM_PROMPT),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# Conversation Prompt
CONV_SYSTEM_PROMPT = """You are a helpful assistant that can answer questions on a wide range of topics."""

conv_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONV_SYSTEM_PROMPT),
        ("human", "{question}")
    ]
)

# Question Rewriter Prompt
REWRITE_SYSTEM_PROMPT = """You are a question re-writer that converts an input question to a better version that is optimized 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
