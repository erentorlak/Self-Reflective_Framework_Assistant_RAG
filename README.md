# Adaptive Framework Support through Self-Reflective Retrieval-Augmented Generation (Self-RAG) System

This project integrates a **Self-RAG** system to enhance adaptive framework support by leveraging internal knowledge bases or performing web searches. The system employs self-evaluation techniques to ensure reliable and accurate outputs for newly created or in-house-developed frameworks, such as software library documentation or APIs.


![Self-RAG System Workflow](assets/routing.png)



---

## Table of Contents
- [Adaptive Framework Support through Self-Reflective Retrieval-Augmented Generation (Self-RAG) System](#adaptive-framework-support-through-self-reflective-retrieval-augmented-generation-self-rag-system)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Challenges and Solutions](#challenges-and-solutions)
    - [Chunking Problem](#chunking-problem)
      - [Solution:](#solution)
    - [Routing Mechanism](#routing-mechanism)

---

## Project Description

The aim of this project is to implement a **Self-RAG system** that utilizes an internal knowledge base or web search for relevant information retrieval. It then self-evaluates the generated results to produce accurate outputs tailored to adaptive frameworks, including software documentation and APIs.


## Challenges and Solutions

### Chunking Problem

When combining code and documentation, chunking presents significant challenges:
- **Large Chunks**: May lead to over-generalized or irrelevant results.
- **Small Chunks**: Can lose important context, disrupting the understanding of logic or documentation.

#### Solution:
- **LLM-Based Summarization**:
  - Process the entire page of content instead of arbitrary splitting.
  - Generate concise summaries (5–7 sentences) and user-query examples.
  - Concatanate summaries and query examples and use it for vector database.

---

### Routing Mechanism

This system dynamically handles decision-driven tasks:
- Routes tasks like web search, document retrieval, and content generation based on user needs.
- Includes iterative steps to refine queries, assess content quality, and end or continue the workflow.


