import torch
import torch.nn as nn

vocab_size = 50
embedding_dim = 8

embedding = nn.Embedding(vocab_size, embedding_dim)

token_ids = torch.tensor([1, 5, 10])

output = embedding(token_ids)

print("Shape:", output.shape)
