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
        self.context_2_query_attention = nn.Sequential(
            nn.Linear(4096, 768),
            nn.LayerNorm(768, eps=1e-4),
        )
        self.query_2_context_attention = nn.Sequential(
            nn.Linear(144, 768),
            nn.LayerNorm(768, eps=1e-4),
        )
        self.ansfranger_ffn = nn.Linear(configuration.hidden_size, 2)

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
        # pooler_output: torch.Size([bs, 768])
        _context_outs = self.transformer(**context_tokenized)["pooler_output"]

        _query_outs = self.dropout(_query_outs)
        _context_outs = self.dropout(_context_outs)

        batch_size = _query_outs.size(0)
        _query_3d = torch.reshape(
            _query_outs, shape=(batch_size, 12, 64)
        )  # torch.Size([2, 12, 64])

        _context_3d = torch.reshape(
            _context_outs, shape=(batch_size, 64, 12)
        )  # torch.Size([2, 64, 12])

        scores = (
            torch.softmax(_query_3d, dim=2) @ torch.softmax(_context_3d, dim=2)
        ).relu() / torch.tensor(
            8, dtype=torch.float
        )  # torch.Size([2, 12, 12])

        weights = torch.softmax(scores, dim=-1)  # torch.Size([2, 12, 12])

        _context_attn_inp = torch.bmm(_context_3d @ weights, _query_3d).reshape(
            batch_size, 64 * 64
        )
        _context_attn_out = F.relu(self.context_2_query_attention(_context_attn_inp))

        _query_attn_inp = torch.bmm(_query_3d, _context_3d @ weights).reshape(
            batch_size, 12 * 12
        )
        _query_attn_out = F.relu(self.query_2_context_attention(_query_attn_inp))

        x = torch.stack(
            (
                _query_attn_out,
                _context_attn_out,
            ),
            dim=1,
        )
        logging.debug(x.size())

        return weights

        """

        _context_pooled_out_attn = torch.bmm(
            _query_pooled_out, _query_pooled_out @ _weights
        )
        logging.debug(f"_context_pooled_out_attn={_context_pooled_out_attn.size()}")
        _context_pooled_out_attn = self.context_2_query_attention(
            _context_pooled_out_attn
        )
        logging.debug(f"_context_pooled_out_attn={_context_pooled_out_attn.size()}")

        _query_pooled_out_attn = torch.bmm(
            _query_pooled_out, _context_pooled_out @ _weights
        )
        logging.debug(f"_query_pooled_out_attn={_query_pooled_out_attn.size()}")
        _query_pooled_out_attn = self.query_2_context_attention(_query_pooled_out_attn)
        logging.debug(f"_query_pooled_out_attn={_query_pooled_out_attn.size()}")

        _qa_attn = _context_pooled_out_attn @ _query_pooled_out_attn
        logging.debug(f"_qa_attn={_qa_attn.size()}")
        _drops_inp = torch.bmm(_context_pooled_out, _qa_attn)
        """
        # _drops_inp = _query_pooled_out.pick() @ _context_pooled_out
        # logging.debug(f"_drops_inp={_drops_inp.size()}")

        _dropout_outs = self.dropout(_context_pooled_out)
        logging.debug(f"_dropout_outs={_dropout_outs.size()}")
        logits = self.ansfranger_ffn(_dropout_outs)
        logging.debug(f"logits={logits.size()}")
        logging.debug(f"logits_reshape={logits.view(2,8,128//8).size()}")

        # start_logits, end_logits = logits.split(1, dim=1)
        # return start_logits, end_logits
        return torch.split(logits, split_size_or_sections=128, dim=1)
