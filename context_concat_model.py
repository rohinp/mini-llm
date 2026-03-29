import torch
import torch.nn as nn
import torch.nn.functional as F

import tokenizer
from tokenizer import vocab_size, decode

# Hyperparameters (feel free to tweak them)
batch_size = 32
sequence_length = 8
learning_rate = 5e-4
max_iters = 5000
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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size :]

            # 🔥 pad if sequence is too short
            if idx_cond.shape[1] < self.block_size:
                pad = torch.zeros(
                    (idx_cond.shape[0], self.block_size - idx_cond.shape[1]),
                    dtype=torch.long,
                    device=idx.device,
                )
                idx_cond = torch.cat((pad, idx_cond), dim=1)

            logits, _ = self(idx_cond)

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = ContextConcatModel(vocab_size, n_embd, block_size)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for step in range(max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}: loss {loss.item():.4f}")


# generate text
model.eval()
context = torch.zeros((1, 1), dtype=torch.long).to(device)
generated = model.generate(context, max_new_tokens=200)

print(decode(generated[0].tolist()))
