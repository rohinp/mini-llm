import torch

# Step 1 — Load Text
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset length:", len(text))

# Step 2 — Build Vocabulary
chars = sorted(list(set(text)))

vocab_size = len(chars)

print("\nVocabulary size:", vocab_size)
print("Vocabulary characters:\n", chars)

"""Important:
This is your character vocabulary."""

# Step 3 — Create Token Mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print("\nExample mappings:")
for i in range(10):
    print(f"{chars[i]} -> {stoi[chars[i]]}")


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


# Step 4 — Encode and Decode Functions
sample = "To be"
encoded = encode(sample)
decoded = decode(encoded)

print("\nSample text:", sample)
print("Encoded:", encoded)
print("Decoded:", decoded)

# Step 5 — Convert Entire Dataset to Tensor
data = torch.tensor(encode(text), dtype=torch.long)

print("\nData tensor shape:", data.shape)
print("First 20 token IDs:", data[:20])
