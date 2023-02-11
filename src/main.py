"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-02-11 01:22:29
"""

import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from densePassageRetrivalModel import (
    DensePassageRetrivalConfiguration,
    DensePassageRetrivalDeepNeuralNetM,
    WorkPieceDomainTokenizer,
)
from src.config import Config


def setups_reproducibility():
    np.random.seed(0)
    seed = torch.Generator().seed()
    torch.manual_seed(seed)
    logging.info(f"Seeding has been set for numpy and pytorch - {seed}")


def load_dataset(training_path: str, validation_path: str):
    training_data = json.load(open(training_path))
    validation_data = json.load(open(validation_path))
    return pd.DataFrame(training_data), pd.DataFrame(validation_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setups_reproducibility()

    logging.info(f"Configuration={Config()}")
    train_dfx, valid_dfx = load_dataset(
        training_path=Config.TRAINING_SET_PATH,
        validation_path=Config.VALIDATION_SET_PATH,
    )
    logging.info(dict(train_size=train_dfx.shape, valid_size=valid_dfx.shape))

    configuration = DensePassageRetrivalConfiguration()
    logging.info(configuration)

    """
    dpr_model = DensePassageRetrivalDeepNeuralNetM(config=configuration)
    logging.info(dpr_model)

    query_tensor = torch.randn(size=(3, 3, 3))
    context_tensor = torch.randn(size=(3, 3, 5))
    logits = dpr_model(query_tensor, context_tensor)
    logging.info(dict(logits=logits, size=logits.size()))
    """

    tokenizer = WorkPieceDomainTokenizer(
        pretraining_folder=Config.PRETRAINING_TOKEN_FOLDER, pretrained=False
    )
    tokenizer.train(files=['main.py'])
    # outs = tokenizer(["dune is not a good person"])
    # logging.info(outs)
