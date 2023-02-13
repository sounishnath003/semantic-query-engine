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
from torch.utils.data import DataLoader

from densePassageRetrivalModel import (
    DensePassageRetrivalConfiguration,
    DensePassageRetrivalDeepNeuralNetM,
    DensePassageRetrivalDocument,
    OpenDomainQuestionAnsweringModel,
    WordPieceDomainTokenizer,
)
from src.config import Config
from src.dataset import Dataset

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

    """
    original_sent = "Ratan tata is the big shot"
    original_sent2 = "Ratan tata owns Tashinq"
    outs = tokenizer(
        original_sent,
        original_sent2,
        truncation="only_second",
        max_length=Config.MAX_SEQUENCE_LENGTH,
        padding="max_length",
        stride=Config.STRIDE_LENGTH,
    )
    retus = tokenizer.decode(outs["input_ids"])
    logging.info(
        dict(
            original=f"{original_sent} {original_sent2}",
            encode=outs,
            back_2_original=retus,
        )
    )
    """

    train_dataset = Dataset(tokenizer=tokenizer, documents=train_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)

    valid_dataset = Dataset(tokenizer=tokenizer, documents=valid_data)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=True)
    # logging.info(train_dataset[1])

    qa_model = OpenDomainQuestionAnsweringModel(configuration=configuration)
    ddict = next(iter(train_dataloader))
    outs = qa_model(**ddict)
    logging.info(ddict)

    """
    # dpr_model = DensePassageRetrivalDeepNeuralNetM(config=configuration)
    query_tensor = torch.randn(size=(3, 3, 3))
    context_tensor = torch.randn(size=(3, 3, 5))
    logits = dpr_model(query_tensor, context_tensor)
    logging.info(dict(logits=logits, size=logits.size()))
    """
