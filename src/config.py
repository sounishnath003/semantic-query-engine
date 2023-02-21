"""
# _* coding: utf8 *_

filename: config.py

@author: sounishnath
createdAt: 2023-02-11 11:55:41
"""

from dataclasses import dataclass
from glob import glob

from transformers import AutoModel, AutoTokenizer, DistilBertForQuestionAnswering


@dataclass
class Config:
    TRAINING_SET_PATH: str = "./data/trainset.json"
    VALIDATION_SET_PATH: str = "./data/validset.json"

    USE_PRETRAINED_TOKENIZER: bool = True
    MAX_SEQUENCE_LENGTH: int = 384
    STRIDE_LENGTH: int = 128
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    ACCUMULATION_STEP = 1
    EPOCHS = 20

    CORPUS_FILES = glob("./corpus/*.txt")
    # PRETRAINING_TOKEN_FOLDER: str = "./pretrained/pretraining_tokenizer_out"
    MODEL_NAME = "distilbert-base-uncased"
    PRETRAINING_TOKEN_FOLDER: str = f"./pretrained/{MODEL_NAME}"
    TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(
        f"./pretrained/{MODEL_NAME}"
    )
    BERT_MODEL: AutoModel = DistilBertForQuestionAnswering.from_pretrained(
        f"./pretrained/{MODEL_NAME}"
    )

    TOKENIZER_CONFIG_DICT = dict(
        add_special_tokens=True,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        stride=STRIDE_LENGTH,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_token_type_ids=True,
    )
