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


class OpenDomainQuestionAnsweringModel(nn.Module):
    def __init__(
        self,
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

        self.transformer = AutoModel.from_pretrained(
            configuration.model_name, config=config
        )
        self.dropout = nn.Dropout(0.20)
        self.ansfranger_ffn1 = nn.Linear(1536, 512)
        self.ansfranger_ffn2 = nn.Linear(1536, 512)

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
        query_tokenized,
        context_tokenized,
        answer_start_index=None,
        answer_end_index=None,
    ):
        """
        1. query and context tokens passed in params encode them using bert encoder
        2. use `attention` to find similarity match query-vectors and context-vectors after step.1
        3. compute `similarity matrix`
        4. combined c2q, q2c attention
        5.
        """

        # pooler_output: torch.Size([bs, 768])
        _query_outs = self.transformer(**query_tokenized)["pooler_output"]
        _query_outs = self.dropout(_query_outs)

        # pooler_output: torch.Size([bs, 768])
        _context_outs = self.transformer(**context_tokenized)["pooler_output"]
        _context_outs = self.dropout(_context_outs)

        _outs = torch.concat([_query_outs, _context_outs], dim=1)

        start_logits = self.ansfranger_ffn1(_outs)
        end_logits = self.ansfranger_ffn2(_outs)
        # start_logits, end_logits = torch.split(logits, split_size_or_sections=1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
