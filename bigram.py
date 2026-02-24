import torch
import torch.nn.functional as F

# Import tokenizer objects
from tokenizer import data, vocab_size, decode

# Step 1 — Build Bigram Count Matrix

bigram_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

# Count occurrences
for i in range(len(data) - 1):
    current_token = data[i].item()
    next_token = data[i + 1].item()
    bigram_counts[current_token, next_token] += 1

print("Bigram matrix shape:", bigram_counts.shape)

# Step 2 — Convert Counts to Probabilities

# Convert to float
bigram_probs = bigram_counts.float()

# Normalize rows to get probabilities
bigram_probs = bigram_probs / bigram_probs.sum(dim=1, keepdim=True)


# Step 3 — Generate Text
def generate(start_token, max_new_tokens=100):
    current = start_token
    generated = [current]

    for _ in range(max_new_tokens):
        probs = bigram_probs[current]
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        current = next_token

    return decode(generated)


# Start from token 0
print(generate(start_token=0, max_new_tokens=200))
