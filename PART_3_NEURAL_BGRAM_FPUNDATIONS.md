# NEURAL BIGRAM MATH FOUNDATIONS

## Part 3 — Logits, Softmax, and Cross Entropy (Deep Understanding)

This section documents the internal mechanics behind:

* Logits
* Softmax
* Sampling
* Cross Entropy
* Numerical Stability

This removes the “black box” feeling.

---

# 1. What Are Logits?

In the Neural Bigram model:

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

This creates a weight matrix:

[
W \in \mathbb{R}^{V \times V}
]

Where:

* ( V = vocab_size )

When a token ID is passed in:

```python
logits = W[token_id]
```

We simply select a row from this matrix.

That row is a vector of length `vocab_size`.

Example:

```
[ 0.23, -1.5, 0.7, 3.1, ..., 0.12 ]
```

These values are called **logits**.

---

## Definition

**Logits = raw, unnormalized scores for each possible next token.**

They are:

* Not probabilities
* Not restricted between 0 and 1
* Can be negative or positive
* Initially random
* Later shaped by training

---

# 2. Where Do Logits Come From?

At initialization:

The embedding table is filled with random values.

So logits start random.

During training:

```python
loss.backward()
optimizer.step()
```

Each logit value is updated using gradient descent:

[
w = w - \eta \cdot \frac{\partial Loss}{\partial w}
]

Over time:

* Correct next-token logits increase
* Incorrect logits decrease

The model learns preference scores.

---

# 3. Why Do We Need Softmax?

Logits are raw scores.

But probabilities must:

[
0 \le p \le 1
]
[
\sum p = 1
]

Softmax converts logits into probabilities.

---

## Softmax Formula

[
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

Where:

* ( z_i ) = logit

---

## Example

Logits:

```
[2.0, 1.0, 0.1]
```

Exponentials:

```
e^2   ≈ 7.39
e^1   ≈ 2.71
e^0.1 ≈ 1.10
```

After normalization:

```
[0.65, 0.24, 0.11]
```

Now we have valid probabilities.

---

# 4. Important Property of Softmax

Softmax depends on **differences**, not absolute values.

[
softmax([100, 99, 98]) = softmax([2, 1, 0])
]

Because:

[
softmax(z) = softmax(z - c)
]

for any constant ( c ).

Only relative gaps matter.

---

# 5. Numerical Stability Problem

Softmax uses exponentials.

Large logits cause overflow:

[
e^{1000} \rightarrow \infty
]

Which leads to:

```
inf / inf → NaN
```

Training crashes.

---

# 6. How Frameworks Fix This

Before exponentiation:

[
z = z - \max(z)
]

Example:

Instead of:

```
[1000, 999, 998]
```

We compute:

```
[0, -1, -2]
```

Exponentials now safe:

```
e^0 = 1
e^-1 ≈ 0.37
e^-2 ≈ 0.13
```

Same probabilities. No overflow.

---

# 7. Why We Do NOT Apply Softmax Before Cross Entropy

We use:

```python
loss = F.cross_entropy(logits, targets)
```

NOT:

```python
probs = softmax(logits)
loss = F.cross_entropy(probs, targets)
```

Why?

Because `F.cross_entropy()`:

* Internally applies **log-softmax**
* Uses a numerically stable implementation
* Avoids computing large exponentials explicitly
* Avoids double-softmax errors

Internally it computes:

[
\text{log_softmax}(logits)
]

then extracts the log-probability of the correct class.

---

# 8. What Is Cross Entropy Doing?

Cross entropy computes:

[
Loss = -\log(P_{correct})
]

If model is confident and correct:

[
P_{correct} \approx 1
]

Loss ≈ 0

If model is uncertain:

[
P_{correct} \approx 0.33
]

Loss is higher.

If model is confidently wrong:

[
P_{correct} \approx 0
]

Loss is very large.

Training pushes:

* Correct logits up
* Incorrect logits down

Increasing the gap between them.

---

# 9. Why Sampling Requires Softmax

Sampling uses:

```python
torch.multinomial(probs, num_samples=1)
```

It requires probabilities.

Without softmax:

* Logits can be negative
* They don’t sum to 1
* They don’t represent probability

Softmax converts raw scores into a valid distribution.

---

# 10. Summary

Logits:

* Raw preference scores
* Direct rows of embedding table
* Learned through gradient descent

Softmax:

* Converts scores → probabilities
* Amplifies differences
* Shift-invariant
* Can overflow without stabilization

Cross Entropy:

* Accepts logits directly
* Uses stable log-softmax internally
* Computes negative log likelihood of correct class

Numerical Stability:

* Large exponentials cause overflow
* Framework subtracts max(logits) before exponentiation
* Prevents NaNs during training

---

You now understand:

* What logits really are
* How they are computed
* Why softmax is needed
* Why softmax can explode
* Why cross_entropy takes logits
* Why numerical stability matters in deep learning

This foundation is critical for understanding transformers next.
