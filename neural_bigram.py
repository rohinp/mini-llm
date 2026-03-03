import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import vocab_size, decode
from tokenizer import train_data, val_data

batch_size = 32
sequence_length = 8
learning_rate = 1e-2
max_iters = 2000


class NeuralBigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Embedding layer: vocab_size → vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: (B, T)
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape

        # reshape for cross entropy
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def get_batch(split):
        data_source = train_data if split == "train" else val_data
        ix = torch.randint(len(data_source) - sequence_length, (batch_size,))
        x = torch.stack([data_source[i : i + sequence_length] for i in ix])
        y = torch.stack([data_source[i + 1 : i + sequence_length + 1] for i in ix])
        return x, y
