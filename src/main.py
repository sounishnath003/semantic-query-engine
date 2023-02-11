"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-02-11 01:22:29
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
import torch.nn.functional as F

from densePassageRetrivalModel import (
    DensePassageRetrivalConfiguration,
    DensePassageRetrivalDeepNeuralNetM,
    DensePassageRetrivalDocument,
    WordPieceDomainTokenizer,
)
from src.config import Config

## GLOBAL PRESETS #########
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG)
## GLOBAL PRESETS #########


def setups_reproducibility():
    np.random.seed(0)
    seed = torch.Generator().seed()
    torch.manual_seed(seed)
    logging.info(f"Seeding has been set for numpy and pytorch - {seed}")


def load_dataset(training_path: str, validation_path: str):
    training_data_json = json.load(open(training_path))
    validation_data_json = json.load(open(validation_path))
    return training_data_json, validation_data_json


def initialize_tokenizer_loaders():
    logging.basicConfig(level=logging.DEBUG)
    try:
        tokenizer = WordPieceDomainTokenizer(
            pretraining_folder=Config.PRETRAINING_TOKEN_FOLDER,
            pretrained=Config.USE_PRETRAINED_TOKENIZER,
        )
        __trained_tokenizer = tokenizer.train(files=Config.CORPUS_FILES)
        return __trained_tokenizer
    except Exception as e:
        logging.error(e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setups_reproducibility()

    logging.info(f"Configuration={Config()}")
    logging.info(f"CORPUS_FILES={Config.CORPUS_FILES}")
    traindata_json, validdata_json = load_dataset(
        training_path=Config.TRAINING_SET_PATH,
        validation_path=Config.VALIDATION_SET_PATH,
    )
    train_data = [DensePassageRetrivalDocument(**dtata) for dtata in traindata_json]
    valid_data = [DensePassageRetrivalDocument(**dtata) for dtata in validdata_json]
    logging.info(dict(train_size=len(train_data), valid_size=len(valid_data)))

    configuration = DensePassageRetrivalConfiguration()
    logging.info(configuration)

    tokenizer = initialize_tokenizer_loaders()
    original_sent = "Ratan tata is a big shot of India."
    outs = tokenizer(original_sent)
    retus = tokenizer.decode(outs["input_ids"])
    logging.info(dict(original=original_sent, encode=outs, back_2_original=retus))

    dpr_model = DensePassageRetrivalDeepNeuralNetM(config=configuration)

    """
    query_tensor = torch.randn(size=(3, 3, 3))
    context_tensor = torch.randn(size=(3, 3, 5))
    logits = dpr_model(query_tensor, context_tensor)
    logging.info(dict(logits=logits, size=logits.size()))
    """
