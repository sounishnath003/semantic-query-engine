"""
# _* coding: utf8 *_

filename: reader_model.py

@author: sounishnath
createdAt: 2023-02-11 01:59:16
"""


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryContextReaderEncoder(nn.Module):
    def __init__(self) -> None:
        super(QueryContextReaderEncoder, self).__init__()

    def forward(self, input: torch.Tensor):
        return input
