"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2023-02-11 18:14:12
"""

import json
import typing
from dataclasses import dataclass

from densePassageRetrivalModel.wordpiece_tokenizer import BertTokenizer


@dataclass
class AnswerType:
    answer: str
    startIndex: int
    endIndex: int


class DensePassageRetrivalDocument:
    def __init__(self, query: str, context: str, answer: typing.Any) -> None:
        self.query = query
        self.context = context
        self.answer = AnswerType(**answer)

    def to_dictionary(self):
        return {
            "__doctype__": DensePassageRetrivalDocument.__name__,
            "query": self.query,
            "context": self.context,
            "answer": self.answer,
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dictionary())
