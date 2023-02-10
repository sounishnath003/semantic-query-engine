"""
# _* coding: utf8 *_

filename: model.py

@author: sounishnath
createdAt: 2023-02-11 01:53:35
"""

import logging
a
import torch
import torch.nn as nn
import torch.nn.functional as F

from densePassageRetrival.config import DensePassageRetrivalConfiguration
from densePassageRetrival.models.reader_model import QuestionReaderEncoder
from densePassageRetrival.models.retrival_model import PassageRetrivalEncoder


class DensePassageRetrivalModel(nn.Module):
    def __init__(self, config: DensePassageRetrivalConfiguration) -> None:
        super(DensePassageRetrivalModel, self).__init__()
        self.config = config
        self.retrival_encoder = PassageRetrivalEncoder()
        self.reader_encoder = QuestionReaderEncoder()
