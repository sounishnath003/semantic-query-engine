"""
# _* coding: utf8 *_

filename: qa_model.py

@author: sounishnath
createdAt: 2023-02-12 20:44:04
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from densePassageRetrivalModel.config import DensePassageRetrivalConfiguration

from src.config import Config


class OpenDomainQuestionAnsweringModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained_model: AutoModel,
        configuration: DensePassageRetrivalConfiguration,
    ) -> None:
        super(OpenDomainQuestionAnsweringModel, self).__init__()
        logging.basicConfig(level=logging.DEBUG)

        self.model_name = configuration.model_name
        self.num_train_steps = configuration.num_train_steps
        self.learning_rate = configuration.learning_rate

        hidden_dropout_prob = 0.1
        layer_norm_eps = 1e-7

        config = AutoConfig.from_pretrained(configuration.model_name)
        config.update(
            {
                "output_hidden_states": False,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": 3,
            }
        )

        self.transformer = pretrained_model
        self.dropout = nn.Dropout(0.20)
        self.start_span_clf = nn.Linear(
            Config.MAX_SEQUENCE_LENGTH, Config.MAX_SEQUENCE_LENGTH
        )
        self.end_span_clf = nn.Linear(
            Config.MAX_SEQUENCE_LENGTH, Config.MAX_SEQUENCE_LENGTH
        )

    def optimizer_scheduler(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = torch.optim.AdamW(optimizer_parameters, lr=self.learning_rate)
        sch = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
        )

        return opt, sch

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        start_positions=None,
        end_positions=None,
    ):
        """
        1. query and context tokens passed in params encode them using bert encoder
        2. use `attention` to find similarity match query-vectors and context-vectors after step.1
        3. compute `similarity matrix`
        4. combined c2q, q2c attention
        5.
        """
        __outs = self.transformer(
            **dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
                return_dict=True,
            )
        )

        start_logits, end_logits = self.dropout(__outs.start_logits), self.dropout(
            __outs.end_logits
        )
        start_logits = self.start_span_clf(start_logits)
        end_logits = self.end_span_clf(end_logits)

        return start_logits, end_logits, __outs.loss
