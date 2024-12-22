# app/graders.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.prompts import (
    grade_documents_prompt,
    hallucination_prompt,
    answer_prompt,
)
from app.models import GradeDocuments, GradeHallucinations, GradeAnswer

class Graders:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        # Grader for Documents
        self.retrieval_grader = (grade_documents_prompt |
                                  self.llm.with_structured_output(GradeDocuments))

        # Grader for Hallucinations
        self.hallucination_grader = (hallucination_prompt |
                                      self.llm.with_structured_output(GradeHallucinations))

        # Grader for Answer
        self.answer_grader = (answer_prompt |
                               self.llm.with_structured_output(GradeAnswer))

    def grade_documents(self, question: str, document: str) -> GradeDocuments:
        return self.retrieval_grader.invoke({"question": question, "document": document})

    def grade_hallucinations(self, documents: str, generation: str) -> GradeHallucinations:
        return self.hallucination_grader.invoke({"documents": documents, "generation": generation})

    def grade_answer(self, question: str, generation: str) -> GradeAnswer:
        return self.answer_grader.invoke({"question": question, "generation": generation})
