"""
# _* coding: utf8 *_

filename: workpiece_tokenizer.py

@author: sounishnath
createdAt: 2023-02-11 13:08:46
"""

import os

from tokenizers import BertWordPieceTokenizer


class WorkPieceDomainTokenizer:
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
        try:
            if self.is_pretrained:
                pass
            else:
                self.__train_tokenizer_step(
                    files,
                    vocab_size,
                    min_frequency,
                    limit_alphabet,
                    wordpieces_prefix,
                    special_tokens,
                )
        except Exception as e:
            print(e)

    def __train_tokenizer_step(
        self,
        corpus,
        vocab_size,
        min_frequency,
        limit_alphabet,
        wordpieces_prefix,
        special_tokens,
    ):
        try:
            self.tokenizer.train(
                corpus,
                vocab_size,
                min_frequency,
                limit_alphabet,
                wordpieces_prefix,
                special_tokens,
            )
            os.mkdir(self.pretrained_folder)
            self.tokenizer.save_model(
                self.pretrained_folder, "pretraining-tokenizer-model"
            )
        except Exception as e:
            return e
