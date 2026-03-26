import torch
import torch.nn as nn
import torch.nn.functional as F

import tokenizer
from tokenizer import vocab_size, decode

# Hyperparameters (feel free to tweak them)
batch_size = 32
sequence_length = 8
learning_rate = 1e-3
max_iters = 2000
eval_interval = 200
eval_iters = 100
n_embd = 32
temperature = 1.5
block_size = 8


def get_batch(split):
    return tokenizer.get_batch_with_block(split, block_size)


class ContextConcatModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.block_size = block_size

        # input size = block_size * n_embd
        self.lm_head = nn.Linear(n_embd * block_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, block_size)

        x = self.token_embedding_table(idx)  # (B, block_size, n_embd)

        # flatten (concatenate)
        B, T, C = x.shape
        x = x.view(B, T * C)  # (B, block_size * n_embd)

        logits = self.lm_head(x)  # (B, vocab_size)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits, targets)

        return logits, loss


model = ContextConcatModel(vocab_size, n_embd, block_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -self.block_size :]  # crop context

        logits, _ = self(idx_cond)

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
