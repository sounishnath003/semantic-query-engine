"""
# _* coding: utf8 *_

filename: test.py

@author: sounishnath
createdAt: 2023-02-23 22:20:34
"""

from src.config import Config
from src.preformats_utils import PreformatDatasetProcessor

if __name__ == "__main__":
    train_gen = PreformatDatasetProcessor.LoadDataset(Config.TRAINING_SET_PATH)
    print("length of train dataset = ", list(train_gen).__len__())

    valid_gen = PreformatDatasetProcessor.LoadDataset(Config.VALIDATION_SET_PATH)
    print("length of valid dataset = ", list(valid_gen).__len__())
