"""
# _* coding: utf8 *_

filename: config.py

@author: sounishnath
createdAt: 2023-02-11 11:55:41
"""

from dataclasses import dataclass
from glob import glob


@dataclass
class Config:
    TRAINING_SET_PATH: str = "./data/trainset.json"
    VALIDATION_SET_PATH: str = "./data/validset.json"
    PRETRAINING_TOKEN_FOLDER: str = "./pretraining_tokenizer_out"
    CORPUS_FILES = glob("./corpus/*.txt")
    USE_PRETRAINED_TOKENIZER: bool = True
    MAX_SEQUENCE_LENGTH: int = 128
    STRIDE_LENGTH: int = 32
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    ACCUMULATION_STEP = 1
    EPOCHS = 3
