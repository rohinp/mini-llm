import torch

batch_size = 2  # numer of sentence
sequence_length = 5  # length of the sentence
embedding_dim = 3  # number of ways a token can be represented.

x = torch.randn(batch_size, sequence_length, embedding_dim)

print("Tensor shape:", x.shape)


print("\nAccess first sentence:")
print(x[0])

print("\nAccess first token of first sentence:")
print(x[0][0])

print("\nAccess one number from that token:")
print(x[0][0][1])
