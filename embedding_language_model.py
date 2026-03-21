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


def get_batch(split):
    return tokenizer.get_batch(split, batch_size, sequence_length)


class EmbeddingLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)  # (B, T, n_embd)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(idx)

                # focus only on last time step
                logits = logits[:, -1, :]  # (B, C)

                probs = F.softmax(logits / temperature, dim=-1)

                idx_next = torch.multinomial(probs, num_samples=1)  # sample

                idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = EmbeddingLanguageModel(vocab_size, n_embd)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# training loop
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()  # clear old gradients
    loss.backward()  # compute gradients
    optimizer.step()  # update weights


model.eval()
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=200)

print(decode(generated[0].tolist()))
