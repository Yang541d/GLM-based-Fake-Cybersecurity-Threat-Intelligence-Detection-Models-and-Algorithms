import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from typing import Optional, Tuple, Union

# ======================================================================
# File: FCTICM/FCTICM-TC.py
# Purpose: Define the BERT-TextCNN detection model.
# Implementation notes:
#   - Requirement 3: BERT parameters are NOT frozen so the encoder is fully fine-tuned.
#   - Paper hyperparameters for detection: filter sizes (2,3,4), number of filters=256, dropout=0.3.
# ======================================================================


class BertTextCNN(nn.Module):
    """
    BERT-TextCNN for binary classification on CTI detection.
    Fulfills Requirement 3 by not freezing any BERT parameters.
    """

    def __init__(self, pretrained_name: str = "google-bert/bert-base-chinese", num_labels: int = 2,
                 filter_sizes=(2, 3, 4), num_filters: int = 256, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        hidden = self.bert.config.hidden_size

        # No freezing here (Requirement 3). Keep BERT trainable.
        # for p in self.bert.parameters():
        #     p.requires_grad = False  # (intentionally not used)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden, out_channels=num_filters, kernel_size=ks)
            for ks in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)  # Paper: 0.3
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        x = last_hidden.permute(0, 2, 1)  # [B, H, L]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(t, kernel_size=t.size(2)).squeeze(2) for t in conv_outs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        logits = self.classifier(cat)
        return logits


def build_model(pretrained_name: str = "google-bert/bert-base-chinese",
                num_labels: int = 2,
                filter_sizes=(2, 3, 4),
                num_filters: int = 256,
                dropout: float = 0.3) -> BertTextCNN:
    """
    Factory to create the model with paper-specified defaults.
    - Filter sizes: (2,3,4)
    - Num filters: 256
    - Dropout: 0.3
    """
    return BertTextCNN(
        pretrained_name=pretrained_name,
        num_labels=num_labels,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        dropout=dropout,
    )