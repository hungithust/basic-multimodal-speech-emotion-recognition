import torch
import torch.utils.checkpoint
from transformers import AutoModel, AutoTokenizer
from torch import nn


DEBERTA_V3_MODEL_NAME = "vinai/phobert-base"


class DebertaV3(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super(DebertaV3, self).__init__()
        self.deberta = AutoModel.from_pretrained(DEBERTA_V3_MODEL_NAME)
        self.mean = nn.AdaptiveAvgPool1d(1)
        self.lm_head = nn.Linear(768, hidden_size)
        self.lm_head_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.cls_head = nn.Linear(hidden_size // 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def freeze_feature_extractor(self):
        self.deberta.freeze_feature_encoder() if hasattr(self.deberta, 'freeze_feature_encoder') else None

    def forward(self, x: torch.Tensor):
        x = self.deberta(x).last_hidden_state
        x = x.permute(0, 2, 1)  # (batch_size, hidden_size, seq_len)
        x = self.lm_head(x)
        x = self.lm_head_2(x)
        x = self.cls_head(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_classes)
        x = self.mean(x)  # (batch_size, num_classes, 1)
        x = x.squeeze(2)  # (batch_size, num_classes)
        x = self.softmax(x)
        return x


def DebertaV3Tokenizer():
    return AutoTokenizer.from_pretrained(DEBERTA_V3_MODEL_NAME)
