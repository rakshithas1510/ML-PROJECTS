"""
src/model.py
------------
Deep Learning models for ticket classification.

Architectures
-------------
BiLSTMClassifier  – bidirectional LSTM with attention pooling  (default)
TextCNNClassifier – multi-kernel 1-D CNN (Kim 2014)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Attention Pooling ────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Soft attention over LSTM hidden states.

    Learns a context vector to weight each time-step's output.
    Returns a weighted sum of shape (batch, hidden_dim).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lstm_out : (batch, seq_len, hidden_dim)
        mask     : (batch, seq_len)  True where token is PAD
        """
        # (batch, seq_len, 1) → (batch, seq_len)
        scores = self.attention(lstm_out).squeeze(-1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)                # (batch, seq_len)
        # Weighted sum
        pooled = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (batch, hidden)
        return pooled


# ─── BiLSTM Classifier ───────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with attention pooling for text classification.

    Architecture
    ------------
    Embedding → BiLSTM (stacked) → Attention Pooling → Dropout → Linear
    """

    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int,
        hidden_dim:    int,
        num_classes:   int,
        num_layers:    int   = 2,
        dropout:       float = 0.3,
        pad_idx:       int   = 0,
        bidirectional: bool  = True,
    ):
        super().__init__()

        self.pad_idx  = pad_idx
        self.hidden_dim = hidden_dim
        self.num_dir  = 2 if bidirectional else 1

        # ── Layers ──────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size    = embedding_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionPooling(hidden_dim * self.num_dir)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * self.num_dir, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[self.pad_idx].zero_()  # pad → zeros
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1 (helps with longer sequences)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len)  integer token indices

        Returns
        -------
        logits : (batch, num_classes)
        """
        pad_mask = x == self.pad_idx                          # (batch, seq_len)
        emb      = self.dropout(self.embedding(x))            # (batch, seq, emb)
        lstm_out, _ = self.lstm(emb)                          # (batch, seq, hid*dirs)
        pooled   = self.attention(lstm_out, pad_mask)         # (batch, hid*dirs)
        out      = self.dropout(pooled)
        logits   = self.fc(out)                               # (batch, num_classes)
        return logits


# ─── TextCNN Classifier ───────────────────────────────────────────────────────

class TextCNNClassifier(nn.Module):
    """
    Kim (2014) – Convolutional Neural Networks for Sentence Classification.

    Uses multiple filter widths and max-over-time pooling.
    """

    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int,
        num_classes:   int,
        num_filters:   int   = 128,
        filter_sizes:  tuple = (2, 3, 4, 5),
        dropout:       float = 0.3,
        pad_idx:       int   = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels  = embedding_dim,
                out_channels = num_filters,
                kernel_size  = fs,
            )
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), num_classes)

        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[pad_idx].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                        # (batch, seq, emb)
        emb = emb.permute(0, 2, 1)                    # (batch, emb, seq) for Conv1d

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                      # (batch, filters, seq-k+1)
            p = c.max(dim=-1).values                   # (batch, filters)
            pooled.append(p)

        cat    = torch.cat(pooled, dim=1)              # (batch, filters*len(sizes))
        out    = self.dropout(cat)
        logits = self.fc(out)
        return logits


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_model(
    architecture: str,
    vocab_size:   int,
    num_classes:  int,
    config:       dict,
) -> nn.Module:
    """
    Instantiate the requested model architecture.

    Parameters
    ----------
    architecture : "bilstm" | "textcnn"
    vocab_size   : vocabulary size (from Vocabulary object)
    num_classes  : number of ticket categories
    config       : full config dict (reads config["model"])
    """
    mcfg = config["model"]
    arch = architecture.lower()

    if arch == "bilstm":
        model = BiLSTMClassifier(
            vocab_size    = vocab_size,
            embedding_dim = mcfg["embedding_dim"],
            hidden_dim    = mcfg["hidden_dim"],
            num_classes   = num_classes,
            num_layers    = mcfg["num_layers"],
            dropout       = mcfg["dropout"],
            bidirectional = mcfg["bidirectional"],
        )
    elif arch == "textcnn":
        model = TextCNNClassifier(
            vocab_size    = vocab_size,
            embedding_dim = mcfg["embedding_dim"],
            num_classes   = num_classes,
            num_filters   = mcfg["hidden_dim"],
            dropout       = mcfg["dropout"],
        )
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose 'bilstm' or 'textcnn'."
        )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {arch.upper()}  |  Trainable params: {total_params:,}")
    return model
