"""
# _* coding: utf8 *_

filename: config.py

@author: sounishnath
createdAt: 2023-02-11 01:38:59
"""

from dataclasses import dataclass


@dataclass
class DensePassageRetrivalConfiguration:
    retrieve_n_passages: int = 3
    max_sequence_length: int = 128
    num_train_epochs: int = 10
    num_train_batch_size: int = 16
    num_valid_batch_size: int = 16
    learning_rate: float = 1e-5
    num_train_steps: int = 100
    model_name: str = "distilbert-base-uncased"
    hidden_size: int = 64
    num_train_steps: int = 100
