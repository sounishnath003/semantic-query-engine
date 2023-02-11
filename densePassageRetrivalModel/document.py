"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2023-02-11 18:14:12
"""

import json

from densePassageRetrivalModel.wordpiece_tokenizer import BertTokenizer


class DensePassageRetrivalDocument:
    def __init__(self, query: str, context: str) -> None:
        self.query = query
        self.context = context
        self.trainable_context = f"[CLS] {self.query} [SEP] {self.context} [SEP]"

    def to_dictionary(self):
        return {
            "__doctype__": DensePassageRetrivalDocument.__name__,
            "query": self.query,
            "context": self.context,
            "trainable_context": self.trainable_context,
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dictionary())
