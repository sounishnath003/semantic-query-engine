"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2023-02-11 18:14:12
"""

import json
import typing
from dataclasses import dataclass

from densePassageRetrivalModel.types import Answer, QuestionAnswerDocument


class DensePassageRetrivalDocument:
    def __init__(self, _document) -> None:
        try:
            document = next(_document)
            self.query = document.query
            self.context = document.context
            self.answer = document.answer
        except Exception as e:
            print(e)

    def to_dictionary(self):
        return {
            "__doctype__": DensePassageRetrivalDocument.__name__,
            "query": self.query,
            "context": self.context,
            "answer": self.answer,
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dictionary())
