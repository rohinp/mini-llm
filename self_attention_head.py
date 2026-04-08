import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)

        # attention scores
        wei = q @ k.transpose(-2, -1)  # (B, T, T)

        wei = wei * (k.shape[-1] ** -0.5)

        # mask future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        v = self.value(x)  # (B, T, H)

        out = wei @ v  # (B, T, H)

        return out
