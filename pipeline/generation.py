"""Structured clinical answer schema and the LLM that produces it.

Uses ChatOpenAI.with_structured_output (OpenAI Structured Outputs, strict json_schema) so the
answer is Pydantic-validated and natively traced in LangSmith. Nodes call .model_dump() to
keep passing a plain dict downstream.
"""
from pydantic import BaseModel

from rag import GENERATION_MODEL, chat_model


class SourceUsed(BaseModel):
    ref: int
    quote: str


class ClinicalAnswer(BaseModel):
    sufficient_information: bool
    answer: str
    sources_used: list[SourceUsed]
    follow_up_questions: list[str]


structured_llm = chat_model(GENERATION_MODEL, temperature=0.2).with_structured_output(
    ClinicalAnswer, method="json_schema", strict=True
)
