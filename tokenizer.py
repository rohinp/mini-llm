import torch
import random

""""-------------------part 0------------------"""

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

"""---------------part 1----------------------"""

# Step 1 Train/val split (90% train, 10% val)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Train data shape:", train_data.shape)
print("Val data shape:", val_data.shape)

# Step 2 — Create get_batch Function
batch_size = 4
sequence_length = 8


def get_batch(split, batch_size_override=None, sequence_length_override=None):
    data_source = train_data if split == "train" else val_data

    bs = batch_size if batch_size_override is None else batch_size_override
    sl = (
        sequence_length
        if sequence_length_override is None
        else sequence_length_override
    )

    # Random starting positions
    ix = torch.randint(len(data_source) - sl, (bs,))

    x = torch.stack([data_source[i : i + sl] for i in ix])
    y = torch.stack([data_source[i + 1 : i + sl + 1] for i in ix])

    return x, y


# Step 3 — Inspect Batch
xb, yb = get_batch("train")

print("Input batch shape:", xb.shape)
print("Target batch shape:", yb.shape)

print("\nFirst input example:")
print(xb[0])
print("\nDecoded input:")
print(decode(xb[0].tolist()))

print("\nDecoded target:")
print(decode(yb[0].tolist()))


# --------- REFER THIS AFTER PART 5 OR LEARNING CONTEXT AWARE MODELS PART --------


def get_batch_with_block(split, block_size, batch_size_override=None):
    data = train_data if split == "train" else val_data
    bs = batch_size if batch_size_override is None else batch_size_override
    ix = torch.randint(len(data) - block_size, (bs,))

    x = torch.stack([data[i : i + block_size] for i in ix])  # (B, block_size)
    y = torch.stack([data[i + block_size] for i in ix])  # (B)

    return x, y


xbb, ybb = get_batch_with_block("train", 8)

print("-------------batch with block shape-----------------")
print("Input batch shape:", xbb.shape)
print("Target batch shape:", ybb.shape)
print("\nFirst input example:")
print(xbb[0])
print("\nDecoded input:")
print(decode(xbb[0].tolist()))

print("\n target:")
print(ybb[0].item())
print("\nDecoded target:")
print(decode([ybb[0].item()]))
