# Adaptive Framework Support through Self-Reflective Retrieval-Augmented Generation (Self-RAG) System

This project integrates a **Self-RAG** system to enhance adaptive framework support by leveraging internal knowledge bases or performing web searches. The system employs self-evaluation techniques to ensure reliable and accurate outputs for newly created or in-house-developed frameworks, such as software library documentation or APIs.

---

## Table of Contents
- [Project Description](#project-description)
- [Challenges and Solutions](#challenges-and-solutions)
  - [Chunking Problem](#chunking-problem)
  - [Routing Mechanism](#routing-mechanism)
- [Example Workflows](#example-workflows)
- [Monitoring and Evaluation](#monitoring-and-evaluation)
- [Project Timeline](#project-timeline)
- [References](#references)

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
  - Store summaries and query examples with metadata in a vector database for efficient retrieval.
- Metadata-based retrieval ensures contextual integrity while preserving detailed content.

---

### Routing Mechanism

This system dynamically handles decision-driven tasks:
- Routes tasks like web search, document retrieval, and content generation based on user needs.
- Includes iterative steps to refine queries, assess content quality, and end or continue the workflow.


