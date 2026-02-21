# Building a Mini LLM — Foundations (Step 0)

This document captures the conceptual foundation before implementing a transformer-based language model from scratch.

The goal is clarity. No magic. No black boxes.

---

# 1. What Is a Language Model?

A language model estimates:

```
P(next token | previous tokens)
```

Example:

Input:

```
"To be or not to"
```

Model predicts:

```
"be"
```

Training objective:

Minimize cross-entropy (negative log-likelihood) between:

* Predicted next-token distribution
* Actual next token (treated as the ground-truth class)

Cross-entropy heavily penalizes confident wrong guesses, so the model learns to put its probability mass on the correct continuation.

---

# 2. What Is a Token?

A **token** is the smallest unit processed by the model.

Depending on tokenizer:

| Type            | Example             |
| --------------- | ------------------- |
| Character-level | `'T'`, `'o'`, `' '` |
| Word-level      | `"To"`, `"be"`      |
| Subword         | `"To"`, `" be"`     |

For our mini model:
👉 We use **character-level tokens**.

Why?

* Tiny vocab (just unique characters) → simple embeddings and no extra tooling.
* Works even if you only have a few kilobytes of text.

Trade-offs to remember:

* Sequences get longer because each word is many characters.
* Semantics are weaker than word/subword tokenizers.

As soon as you outgrow toy datasets, you typically switch to subword tokenizers (BPE, WordPiece, SentencePiece) to shrink sequence length without exploding vocab size.

Example:

Text:

```
"I love AI"
```

Tokens:

```
['I', ' ', 'l', 'o', 'v', 'e', ' ', 'A', 'I']
```

---

# 3. Vocabulary Size vs Sequence Length

These are completely different concepts.

---

## Vocabulary Size

Total number of unique tokens in dataset.

Example:

If Tiny Shakespeare contains 65 unique characters:

[
\text{vocab_size} = 65
]

This is fixed after analyzing dataset.

---

## Sequence Length

How many tokens the model sees at once.

Example:

[
\text{sequence_length} = 128
]

That means:

The model receives 128 consecutive tokens as input.

This also defines the **context window**.

---

## Visual Comparison

Vocabulary:

```
Total possible symbols: 65
```

Sequence:

```
[ T o   b e   o r   n o t ... ]  (128 tokens long)
```

Vocabulary = possible choices
Sequence length = how many tokens we feed at once

They are independent.

---

# 4. Batch Size

Batch size = number of sequences processed in parallel.

If:

[
\text{batch_size} = 32
]

and

[
\text{sequence_length} = 128
]

Then tensor shape:

```
(32, 128)
```

Meaning:

* 32 independent sequences
* Each 128 tokens long

Tokens processed per step = `batch_size × sequence_length` → here `32 × 128 = 4096` tokens. That total, along with `embedding_dimension`, mostly dictates VRAM usage and compute time.

### Important:

Batch items do NOT see each other.
They are processed independently.

---

# 5. Core Tensor Shape in Transformers

Most common shape:
[Shape demo](part0/shape_demo.py)
[Tensor demo](part0/tensor_basics.py)

```
(batch_size, sequence_length, embedding_dimension)
```

Example:

```
(32, 128, 64)
```

Meaning:

* 32 sequences
* 128 tokens each
* Each token represented by 64 numbers

---

## Visual Breakdown

Think of it like:

```
Batch 0:
    Token 0 → [64 numbers]
    Token 1 → [64 numbers]
    ...
Batch 1:
    Token 0 → [64 numbers]
```

---

# 6. What Is an Embedding?

Tokens start as integers:

```
'T' → 19
'o' → 14
```

Embedding layer converts:

```
19 → [0.12, -0.44, 0.91, ...]
```

If:

[
\text{embedding_dimension} = 64
]

Each token becomes a 64-length vector.

---

## Conceptually

Embedding = learned feature space.

Higher dimension →
More expressive power
More compute cost

---

# 7. Why We Break a Novel Into Chunks

We treat the entire novel as one long stream:

```
[ t0, t1, t2, t3, t4, ... ]
```

Then create training samples:

If sequence_length = 4

```
[t0, t1, t2, t3]
[t1, t2, t3, t4]
[t2, t3, t4, t5]
```

Targets are just the same windows shifted left by one token:

| Input tokens      | Target tokens     |
| ----------------- | ----------------- |
| `[t0, t1, t2, t3]` | `[t1, t2, t3, t4]` |
| `[t1, t2, t3, t4]` | `[t2, t3, t4, t5]` |

During training we feed the input window and ask the model to predict the target window one token at a time—this is the next-token objective in action.

This is called sliding window training.

Why not entire novel?

Because attention complexity is:

[
O(n^2)
]

If sequence length doubles,
computation roughly quadruples.

---

# 8. What Is Attention Computation?

Inside one sequence of length N:

Each token compares with every other token.

Attention matrix size:

[
N \times N
]

Example:

Sequence length = 4

```
Tokens: [A, B, C, D]
```

Attention matrix:

```
        A    B    C    D
A     [ •    •    •    • ]
B     [ •    •    •    • ]
C     [ •    •    •    • ]
D     [ •    •    •    • ]
```

Each row = how much that token cares about others.

If sequence length = 128:

Attention matrix = 128 × 128

This is why attention is quadratic.

---

# 9. What Is a Gradient?

Suppose we have a loss function:

[
L(w)
]

Gradient:

[
\frac{dL}{dw}
]

It tells us:

"If I slightly change w, how does loss change?"

---

## Example

[
f(x) = x^2
]

Derivative:

[
\frac{df}{dx} = 2x
]

If:

[
x = 3
]

Then:

[
\text{gradient} = 6
]

Meaning:
Increasing x increases loss.

---

# 10. Why Move Opposite to Gradient?

Gradient points toward steepest increase.

To reduce loss:

[
w_{\text{new}} = w - \eta \cdot \frac{dL}{dw}
]

Where:

* ( \eta ) = learning rate

If gradient is positive:
→ decrease weight

If gradient is negative:
→ increase weight

---

# 11. What Does backward() Do?

Calling:

```
loss.backward()
```

Computes:

[
\frac{\partial L}{\partial w_1},
\frac{\partial L}{\partial w_2},
\frac{\partial L}{\partial w_3},
...
]

Gradients are stored in:

```
parameter.grad
```

Important:

`.backward()` does NOT update weights.

Optimizer step updates weights.

Typical training loop order:

```python
for batch in dataloader:
    optimizer.zero_grad()      # clear stale gradients
    logits = model(batch)      # forward pass
    loss = loss_fn(logits, targets)
    loss.backward()            # compute gradients
    optimizer.step()           # apply parameter update
```

Zeroing grads first avoids mixing gradient information between batches.

---

# 12. Learning Rate and Stability

Update rule:

[
w = w - \eta \cdot \text{gradient}
]

If:

* Gradient large
* Learning rate large

Then update is huge.

Possible outcomes:

* Overshooting minimum
* Instability
* Divergence
* Loss becomes NaN

Training stability requires balance.

---

# Core Mental Model

Loss surface = landscape
Weights = position
Gradient = compass pointing uphill
Optimizer step = move downhill

Training = many small downhill steps.

---

# Conceptual Map So Far

You now understand:

* Tokens
* Vocabulary size
* Sequence length
* Batch size
* Embedding dimension
* Attention scaling
* Gradients
* Backpropagation
* Learning rate effects

This foundation ensures that when we implement the transformer, nothing feels magical.

Everything will be mechanical.

> Courtesy Chatgpt, Reviewer Codex and me.
