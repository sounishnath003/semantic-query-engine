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


class PassageRetrivalEncoder(nn.Module):
    def __init__(self) -> None:
        super(PassageRetrivalEncoder, self).__init__()
