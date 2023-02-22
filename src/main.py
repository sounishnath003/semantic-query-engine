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
from src.engine import trainloop
from src.preformats_utils import PreformatDatasetProcessor

## GLOBAL PRESETS #########
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG)
## GLOBAL PRESETS #########


def setups_reproducibility():
    np.random.seed(0)
    seed = torch.Generator().seed()
    torch.manual_seed(seed)
    logging.info(f"Seeding has been set for numpy and pytorch - {seed}")


def initialize_tokenizer_loaders():
    logging.basicConfig(level=logging.DEBUG)
    """
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
    """
    return Config.TOKENIZER


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setups_reproducibility()

    logging.info(f"Configuration={Config()}")
    logging.info(f"CORPUS_FILES={Config.CORPUS_FILES}")

    raw_train_data = PreformatDatasetProcessor.LoadDataset(
        filepath=Config.TRAINING_SET_PATH
    )
    raw_valid_data = PreformatDatasetProcessor.LoadDataset(
        filepath=Config.VALIDATION_SET_PATH
    )

    train_data = [DensePassageRetrivalDocument(dtata) for dtata in raw_train_data]
    valid_data = [DensePassageRetrivalDocument(dtata) for dtata in raw_valid_data]
    logging.info(dict(train_size=len(train_data), valid_size=len(valid_data)))

    configuration = DensePassageRetrivalConfiguration()
    logging.info(configuration)

    tokenizer = initialize_tokenizer_loaders()

    train_dataset = Dataset(tokenizer=tokenizer, documents=train_data, is_eval=False)
    valid_dataset = Dataset(tokenizer=tokenizer, documents=valid_data, is_eval=True)

    qa_model = OpenDomainQuestionAnsweringModel(
        model_name=Config.MODEL_NAME,
        pretrained_model=Config.BERT_MODEL,
        configuration=configuration,
    )

    tez_configuration = tez.TezConfig(
        device="cpu",
        training_batch_size=Config.TRAIN_BATCH_SIZE,
        validation_batch_size=Config.VALID_BATCH_SIZE,
        epochs=Config.EPOCHS,
        gradient_accumulation_steps=Config.ACCUMULATION_STEP,
        clip_grad_norm=1.0,
        step_scheduler_after="batch",
    )
    qa_model = tez.Tez(
        model=qa_model,
        config=tez_configuration,
        num_train_steps=(
            len(train_dataset)
            / Config.TRAIN_BATCH_SIZE
            / (Config.ACCUMULATION_STEP * Config.EPOCHS)
        ),
        num_valid_steps=(
            len(valid_dataset)
            / Config.TRAIN_BATCH_SIZE
            / (Config.ACCUMULATION_STEP * Config.EPOCHS)
        ),
    )
    qa_model.fit(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )
    qa_model.save("model.bin")

    """
    # dpr_model = DensePassageRetrivalDeepNeuralNetM(config=configuration)
    query_tensor = torch.randn(size=(3, 3, 3))
    context_tensor = torch.randn(size=(3, 3, 5))
    logits = dpr_model(query_tensor, context_tensor)
    logging.info(dict(logits=logits, size=logits.size()))
    """
