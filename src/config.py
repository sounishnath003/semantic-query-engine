"""
# _* coding: utf8 *_

filename: config.py

@author: sounishnath
createdAt: 2023-02-11 11:55:41
"""

from dataclasses import dataclass


@dataclass
class Config:
    TRAINING_SET_PATH: str = "./data/trainset.json"
    VALIDATION_SET_PATH: str = "./data/validset.json"
    PRETRAINING_TOKEN_FOLDER: str = "./pretraining_tokenizer_out"
