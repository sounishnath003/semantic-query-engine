"""
# _* coding: utf8 *_

filename: bertmodel.py

@author: sounishnath
createdAt: 2023-02-11 12:50:52
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
