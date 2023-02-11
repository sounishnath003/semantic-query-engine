"""
# _* coding: utf8 *_

filename: retrival_model.py

@author: sounishnath
createdAt: 2023-02-11 01:56:54
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class PassageContextRetrivalEncoder(nn.Module):
    def __init__(self) -> None:
        super(PassageContextRetrivalEncoder, self).__init__()

    def forward(self, input: torch.Tensor):
        return input
