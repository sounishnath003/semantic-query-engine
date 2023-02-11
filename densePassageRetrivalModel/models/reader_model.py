"""
# _* coding: utf8 *_

filename: reader_model.py

@author: sounishnath
createdAt: 2023-02-11 01:59:16
"""


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class QueryContextReaderEncoder(nn.Module):
    def __init__(
        self, model_name: str, num_train_steps: int, learning_rate: float
    ) -> None:
        super(QueryContextReaderEncoder, self).__init__()
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.learning_rate = learning_rate

        hidden_dropout_prob = 0.1
        layer_norm_eps = 1e-7

        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": 2,
            }
        )

        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
    ):
        _transformer_out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        _pooled_out = _transformer_out.pooler_output
        _dropout_outs = self.dropout(_pooled_out)
        return _dropout_outs
