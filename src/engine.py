"""
# _* coding: utf8 *_

filename: engine.py

@author: sounishnath
createdAt: 2023-02-16 02:06:31
"""

import logging

import tez
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from densePassageRetrivalModel.wordpiece_tokenizer import AutoTokenizer
from src.config import Config


def trainloop(model: nn.Module, tokenizer: AutoTokenizer, dataloader, config: Config):
    lossfn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(config.EPOCHS):
        eloss = 0.0
        eouts = None
        for bid, ddata in tqdm(enumerate(dataloader), total=len(dataloader)):
            opt.zero_grad()
            model(**ddata)
            start = ddata["start_positions"]
            end = ddata["end_positions"]
            startloss = lossfn(start, ddata["start_positions"])
            endloss = lossfn(end, ddata["end_positions"])
            tloss = startloss + endloss

            tloss.backward()
            opt.step()

            eloss += tloss.item()

        logging.info(
            "Epoch [{}/{}], Loss: {:.4f}".format(
                epoch + 1, Config.EPOCHS, eloss / len(dataloader)
            )
        )

        if epoch % 4 == 0:
            pstart = torch.argmax(start, dim=1)
            pend = torch.argmax(end, dim=1)
            logging.info(
                {
                    "originalStart": torch.argmax(ddata["start_positions"], dim=1),
                    "originalEnd": torch.argmax(ddata["end_positions"], dim=1),
                    "predictedStart": pstart,
                    "predictedEnd": pend,
                }
            )
