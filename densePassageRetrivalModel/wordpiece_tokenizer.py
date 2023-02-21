"""
# _* coding: utf8 *_

filename: workpiece_tokenizer.py

@author: sounishnath
createdAt: 2023-02-11 13:08:46
"""

import os

from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer

from densePassageRetrivalModel.config import DensePassageRetrivalConfiguration


class WordPieceDomainTokenizer:
    def __init__(self, pretraining_folder: str, pretrained=False) -> None:
        self.pretrained_folder = pretraining_folder
        self.is_pretrained = pretrained
        self.tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True,
        )

    def train(
        self,
        files: list,
        vocab_size=10_000,
        min_frequency=3,
        limit_alphabet=1000,
        wordpieces_prefix="##",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    ):
        if self.is_pretrained:
            return self.__load_pretrained_tokenizer()
        else:
            __trained_tokenizer = self.__train_tokenizer_step(
                files=files,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                limit_alphabet=limit_alphabet,
                wordpieces_prefix=wordpieces_prefix,
                special_tokens=special_tokens,
            )
            return __trained_tokenizer

    def __train_tokenizer_step(
        self,
        files,
        vocab_size,
        min_frequency,
        limit_alphabet,
        wordpieces_prefix,
        special_tokens,
    ):
        try:
            self.tokenizer.train(
                files=files,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                limit_alphabet=limit_alphabet,
                wordpieces_prefix=wordpieces_prefix,
                special_tokens=special_tokens,
            )
            os.mkdir(self.pretrained_folder)
            self.tokenizer.save_model(self.pretrained_folder)
            return self.__load_pretrained_tokenizer()
        except Exception as e:
            print(e)
            return None

    def __load_pretrained_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.pretrained_folder)
