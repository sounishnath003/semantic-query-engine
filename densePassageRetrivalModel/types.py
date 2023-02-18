"""
# _* coding: utf8 *_

filename: types.py

@author: sounishnath
createdAt: 2023-02-19 01:30:28
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Answer:
    text: str
    answer_start: int

    def __init__(self, text: str, answer_start: int = -1, **kwargs) -> None:
        self.text = text
        self.answer_start = answer_start


@dataclass
class QA:
    question: str
    id: str
    answers: List[Answer]

    def __init__(
        self,
        question: str,
        id: str,
        answers: List[Answer],
        is_impossible: Optional[bool],
        **kwargs
    ) -> None:
        self.question = question
        self.id = id
        self.answers = answers


@dataclass
class Paragraph:
    qas: List[QA]
    context: str

    def __init__(self, qas: List[QA], context: str) -> None:
        self.qas = qas
        self.context = context


@dataclass
class Datum:
    title: str
    paragraphs: List[Paragraph]

    def __init__(self, title: str, paragraphs: List[Paragraph]) -> None:
        self.title = title
        self.paragraphs = paragraphs


@dataclass
class QuestionAnswerDatasetType:
    version: str
    data: List[Datum]

    def __init__(self, version: str, data: List[Datum]) -> None:
        self.version = version
        self.data = data


@dataclass
class QuestionAnswerDocument:
    context: str
    query: str
    answer: Answer

    def __init__(self, context: str, query: str, answer: Answer) -> None:
        self.context = context
        self.query = query
        self.answer = answer
