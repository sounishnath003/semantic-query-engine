"""
# _* coding: utf8 *_

filename: qa_model.py

@author: sounishnath
createdAt: 2023-02-12 20:44:04
"""

import logging

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
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": 2,
            }
        )

        self.transformer = AutoModel.from_pretrained(
            configuration.model_name, config=config
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.context_2_query_attention = nn.ModuleList(
            [
                nn.Linear(configuration.hidden_size, configuration.hidden_size),
                nn.LayerNorm(configuration.hidden_size),
            ]
        )
        self.query_2_context_attention = nn.ModuleList(
            [
                nn.Linear(configuration.hidden_size, configuration.hidden_size),
                nn.LayerNorm(configuration.hidden_size),
            ]
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
        _query_transformer_out = self.transformer(**query_tokenized)
        _query_pooled_out = _query_transformer_out.pooler_output
        logging.debug(f"_query_pooled_out={_query_pooled_out.size()}")

        _context_transformer_out = self.transformer(**context_tokenized)
        _context_pooled_out = _context_transformer_out.pooler_output
        logging.debug(f"_context_pooled_out={_context_pooled_out.size()}")

        _softmaxed_query_distb = torch.softmax(_query_pooled_out, dim=1)
        _softmaxed_context_distb = torch.softmax(_context_pooled_out, dim=1)
        scores = torch.bmm(
            _softmaxed_context_distb, _softmaxed_query_distb.transpose(1, 2)
        ) / torch.tensor(8, dtype=torch.float)
        logging.debug(f"scores={scores.size()}")
        _weights = torch.softmax(scores, dim=-1)
        logging.debug(f"_weights={_weights.size()}")

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
        logging.debug(f"_drops_inp={_drops_inp.size()}")

        _dropout_outs = self.dropout(_drops_inp)
        logging.debug(f"_dropout_outs={_dropout_outs.size()}")
        logits = self.ansfranger_ffn(_dropout_outs)
        logging.debug(f"logits={logits.size()}")

        start_logits, end_logits = logits.split(1, dim=1)
        return start_logits, end_logits
