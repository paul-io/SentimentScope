import math
import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.Q = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.dropout = nn.Dropout(config["dropout_rate"])

        mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        q, k, v = self.Q(x), self.K(x), self.V(x)

        scores = q @ k.transpose(1, 2)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        scores = scores / math.sqrt(k.shape[-1])
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        return scores @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.heads   = nn.ModuleList([AttentionHead(config) for _ in range(config["heads_num"])])
        self.linear  = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.linear(x))


class FeedForward(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config["dropout_rate"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.attention    = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config["d_embed"])
        self.norm2 = nn.LayerNorm(config["d_embed"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class SentimentTransformer(nn.Module):
    """
    A transformer built from scratch for binary sentiment classification.

    Architecture:
      - Token + positional embeddings
      - N stacked transformer blocks (multi-head attention + feed-forward)
      - Mean pooling across the sequence dimension
      - Linear classification head
    """

    def __init__(self, config: dict):
        super().__init__()
        self.token_embedding    = nn.Embedding(config["vocabulary_size"], config["d_embed"])
        self.position_embedding = nn.Embedding(config["context_size"],    config["d_embed"])
        self.blocks     = nn.Sequential(*[TransformerBlock(config) for _ in range(config["layers_num"])])
        self.norm       = nn.LayerNorm(config["d_embed"])
        self.classifier = nn.Linear(config["d_embed"], config["num_classes"], bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)

        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)