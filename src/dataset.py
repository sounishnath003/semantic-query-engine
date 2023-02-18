"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2023-02-13 21:26:26
"""

import typing

import torch
from transformers import BertTokenizer

from densePassageRetrivalModel import DensePassageRetrivalDocument
from src.config import Config


class Dataset:
    def __init__(
        self,
        tokenizer: BertTokenizer,
        documents: typing.List[DensePassageRetrivalDocument],
    ) -> None:
        self.tokenizer = tokenizer
        self.document = documents

    def __len__(self):
        return len(self.document)

    def __getitem__(self, item: int):
        ocontext = self.document[item].context
        oquery = self.document[item].query

        query_tokenized = self.get_query_tokenized(oquery)
        context_tokenized = self.get_context_tokenized(ocontext, oquery)
        answer = self.document[item].answer

        # create the start and end position labels
        start_label = torch.zeros(Config.MAX_SEQUENCE_LENGTH, dtype=torch.float)
        end_label = torch.zeros(Config.MAX_SEQUENCE_LENGTH, dtype=torch.float)
        start_label[answer.startIndex] = 1.0
        end_label[answer.endIndex] = 1.0

        return {
            "query_tokenized": query_tokenized,
            "context_tokenized": context_tokenized,
            "answer_start_index": torch.tensor(start_label, dtype=torch.float),
            "answer_end_index": torch.tensor(end_label, dtype=torch.float),
        }

    def get_context_tokenized(self, ocontext, oquery):
        context = self.tokenizer.encode_plus(
            oquery,
            ocontext,
            truncation="only_second",
            padding="max_length",
            pad_to_max_length=True,
            add_special_tokens=True,
            max_length=Config.MAX_SEQUENCE_LENGTH,
            stride=Config.STRIDE_LENGTH,
        )
        context_tokenized = {
            "input_ids": torch.tensor(context["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(context["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(context["token_type_ids"], dtype=torch.long),
        }

        return context_tokenized

    def get_query_tokenized(self, oquery):
        query = self.tokenizer.encode_plus(
            oquery,
            None,
            padding="max_length",
            pad_to_max_length=True,
            add_special_tokens=True,
            max_length=64,
        )
        query_tokenized = {
            "input_ids": torch.tensor(query["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(query["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(query["token_type_ids"], dtype=torch.long),
        }

        return query_tokenized
