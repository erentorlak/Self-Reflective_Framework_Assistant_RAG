# app/chains.py
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from app.prompts import (
    route_prompt,
    grade_documents_prompt,
    hallucination_prompt,
    answer_prompt,
    conv_prompt,
    re_write_prompt
)
from app.models import RouteQuery
from app.graders import Graders

class Chains:
    def __init__(self, llm: ChatOpenAI, graders: Graders):
        self.llm = llm
        self.graders = graders

        # Router Chain
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.question_router = route_prompt | self.structured_llm_router

        # RAG Chain
        self.rag_prompt = hub.pull("heyheyerent/erent_promptv1")
        #self.rag_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.rag_chain = self.rag_prompt | self.rag_llm | StrOutputParser()

        # Conversation Chain
        self.conv_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.conversation_chain = conv_prompt | self.conv_llm | StrOutputParser()

        # Question Rewriter Chain
        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()
