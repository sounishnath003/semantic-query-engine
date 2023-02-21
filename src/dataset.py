"""
# _* coding: utf8 *_

filename: dataset.py

@author: sounishnath
createdAt: 2023-02-13 21:26:26
"""

import typing

import torch
from transformers import BertTokenizer

from densePassageRetrivalModel import Answer, DensePassageRetrivalDocument
from src.config import Config


class Dataset:
    def __init__(
        self,
        tokenizer: BertTokenizer,
        documents: typing.List[DensePassageRetrivalDocument],
        is_eval=False,
    ) -> None:
        self.tokenizer = tokenizer
        self.document = documents
        self.is_eval = is_eval

    def __len__(self):
        return len(self.document)

    def __getitem__(self, item: int):
        ocontext = self.document[item].context
        oquery = self.document[item].query

        encoded_inputs = self.tokenizer.encode_plus(
            oquery, ocontext, **Config.TOKENIZER_CONFIG_DICT
        )
        answer: Answer = self.document[item].answer

        if self.is_eval:
            start_label, end_label = self.__build_answer_start_end_ranges(
                inputs=encoded_inputs, answer=answer
            )
            encoded_inputs["start_positions"] = torch.tensor(
                start_label, dtype=torch.long
            )
            encoded_inputs["end_positions"] = torch.tensor(end_label, dtype=torch.long)

        return encoded_inputs

    def __build_answer_start_end_ranges(self, inputs, answer: Answer):
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        start_positions = 0
        end_positions = 0

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            start_char = answer.answer_start
            end_char = answer.answer_start + len(answer.text)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if (
                offset[context_start][0] > start_char
                or offset[context_end][1] < end_char
            ):
                start_positions = 0
                end_positions = 0
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions = idx - 1

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions = idx + 1

        return [start_positions], [end_positions]

    def get_query_context_tokenized(self, ocontext, oquery):
        context = self.tokenizer.encode_plus(
            oquery, ocontext, **Config.TOKENIZER_CONFIG_DICT
        )

        context_tokenized = {
            "input_ids": torch.tensor(context["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(context["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(context["token_type_ids"], dtype=torch.long),
        }

        return context_tokenized

    def get_query_tokenized(self, oquery):
        query = self.tokenizer.encode_plus(oquery, None, **Config.TOKENIZER_CONFIG_DICT)
        query_tokenized = {
            "input_ids": torch.tensor(query["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(query["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(query["token_type_ids"], dtype=torch.long),
        }

        return query_tokenized
