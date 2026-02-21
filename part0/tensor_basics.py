import torch

# Scalar
a = torch.tensor(5)
print("Scalar:", a)
print("Shape:", a.shape)

# Vector
v = torch.tensor([1, 2, 3])
print("\nVector:", v)
print("Shape:", v.shape)

# Matrix
m = torch.tensor([[1, 2], [3, 4]])
print("\nMatrix:", m)
print("Shape:", m.shape)

# Random tensor
r = torch.randn(3, 4)
print("\nRandom Tensor:")
print(r)
print("Shape:", r.shape)
