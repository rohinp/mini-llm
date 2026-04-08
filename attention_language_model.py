from self_attention_head import SelfAttentionHead
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


class AttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.sa_head = SelfAttentionHead(n_embd, n_embd, block_size)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        x = self.token_embedding_table(idx)  # (B, T, C)

        x = self.sa_head(x)  # (B, T, C)

        logits = self.lm_head(x)  # (B, T, vocab)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss


model = AttentionLanguageModel(vocab_size, n_embd, block_size)
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
