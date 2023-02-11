"""
# _* coding: utf8 *_

filename: model.py

@author: sounishnath
createdAt: 2023-02-11 01:53:35
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from densePassageRetrivalModel.config import DensePassageRetrivalConfiguration
from densePassageRetrivalModel.models.reader_model import QueryContextReaderEncoder
from densePassageRetrivalModel.models.retrival_model import (
    PassageContextRetrivalEncoder,
)


class DensePassageRetrivalDeepNeuralNetM(nn.Module):
    def __init__(self, config: DensePassageRetrivalConfiguration) -> None:
        super(DensePassageRetrivalDeepNeuralNetM, self).__init__()
        self.config = config
        self.passage_retrival_ranker = PassageContextRetrivalEncoder(
            model_name="bert-base-uncased",
            num_train_steps=config.num_train_steps,
            learning_rate=config.learning_rate,
        )
        self.query_reader_ranker = QueryContextReaderEncoder(
            model_name="bert-base-uncased",
            num_train_steps=config.num_train_steps,
            learning_rate=config.learning_rate,
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor, is_training=True):
        query_reader_encodings = self.query_reader_ranker(query)
        passage_retrival_encodings = self.passage_retrival_ranker(context)
        similarity = self.scaled_similary_computation(
            query_reader_encodings, passage_retrival_encodings
        )
        return similarity

    def scaled_similary_computation(
        self, query_encodings: torch.Tensor, passage_encodings: torch.Tensor
    ) -> torch.Tensor:
        query_dim_k0 = torch.tensor(query_encodings.size(-1), dtype=torch.int)
        transposed = query_encodings.transpose(dim0=1, dim1=2)
        _similarity = torch.bmm(
            transposed, passage_encodings
        )  # / torch.sqrt(query_dim_k0)
        return _similarity
