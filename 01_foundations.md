# Building a Mini LLM — Foundations (Step 0)

This section builds the intuition you need before writing any model code.

You don’t need to understand everything on the first read.
👉 If something feels unclear, continue to the Bigram Model — things will click once you see code.

---

## Why Start with a Bigram Model?

Because it’s the simplest possible language model.

It helps you understand:

* How models learn patterns
* How predictions are made
* What training actually does

We’ll start simple → then gradually build toward a transformer.

---

## What to Expect

* Minimal math
* Strong intuition
* Direct connection to code

**Goal: No magic. No black boxes.**

---

# 1. What Is a Language Model?

As a developer, think of a language model as a function:

```python
next_token = model(previous_tokens)
```

It takes a sequence of tokens as input and predicts what comes next.

More formally:

$$
P(\text{next token} \mid \text{previous tokens})
$$

In simple terms:

👉 “Given what I’ve seen so far, what is the most likely next token?”

---

### Example

Input:

```
"To be or not to"
```

Prediction:

```
"be"
```

---

## Two Phases of a Model

1. **Training**
   The model learns patterns from data.

2. **Inference**
   The model uses what it learned to generate predictions.

---

# 2. What Is a Token?

A **token** is the smallest unit processed by the model.

Before anything happens, text is broken into tokens.

---

### Types of Tokenization

| Type            | Example             |
| --------------- | ------------------- |
| Character-level | `'T'`, `'o'`, `' '` |
| Word-level      | `"To"`, `"be"`      |
| Subword         | `"To"`, `" be"`     |

---

### In This Project

We use **character-level tokens**.

Why?

* Very small vocabulary
* Simple to implement
* Works with tiny datasets

---

### Trade-offs

* Longer sequences
* Weaker understanding of meaning

---

### Example

Text:

```
"I love AI"
```

Tokens:

```
['I', ' ', 'l', 'o', 'v', 'e', ' ', 'A', 'I']
```

---

### Why This Matters (Real World)

* Token count affects API cost
* Context limits depend on tokens
* Tokenization impacts model quality

---

# 3. Vocabulary Size vs Sequence Length

These are two very different concepts.

---

## Vocabulary Size

Total number of unique tokens.

Example:

$$
vocab_size = 65
$$

👉 All possible symbols the model knows.

---

## Sequence Length

How many tokens the model sees at once.

Example:

$$
sequence_length = 128
$$

👉 How much context the model can use.

---

### Mental Model

* Vocabulary → all possible words/characters
* Sequence length → how much the model “remembers”

---

# 4. Batch Size

Batch size = number of sequences processed together.

If:

$$
batch_size = 32,\quad sequence_length = 128
$$

Then:

```
(32, 128)
```

---

### Meaning

* 32 sequences
* Each 128 tokens long

---

### Why This Matters

* Larger batch → faster training (needs more memory)
* Smaller batch → slower but safer

---

## Important

Batch items do NOT see each other.
They are processed independently.

---

# 5. Tensors (Core Data Structure)

A **tensor** is just a multi-dimensional array.

---

### Think of it like:

* 1D → list
* 2D → matrix
* 3D → batch of matrices

---

### Common Shape in Models

```
(batch_size, sequence_length, embedding_dimension)
```

Example:

```
(32, 128, 64)
```

---

### Meaning

* 32 sequences
* 128 tokens each
* Each token → 64 numbers

---

# 6. What Is an Embedding?

Tokens start as integers:

```
'T' → 19
'o' → 14
```

The model cannot understand raw numbers like this.

So we convert them into vectors:

```
19 → [0.12, -0.44, 0.91, ...]
```

---

### Definition

Embedding = learned numerical representation of a token.

---

### Why This Matters

👉 Converts symbols into something the model can learn from

---

### Intuition

Think of embeddings as coordinates in space:

* Similar tokens → closer together
* Different tokens → farther apart

---

# 7. How Training Data Is Prepared

We treat the dataset as one long sequence:

```
[t0, t1, t2, t3, t4, ...]
```

---

### Create Input Sequences

If `sequence_length = 4`:

```
[t0, t1, t2, t3]
[t1, t2, t3, t4]
[t2, t3, t4, t5]
```

---

### Targets (Shifted by One)

| Input (x)          | Target (y)         |
| ------------------ | ------------------ |
| `[t0, t1, t2, t3]` | `[t1, t2, t3, t4]` |

---

### What Are We Teaching?

👉 “Given this context, what comes next?”

Repeated thousands of times → model learns patterns.

---

# 8. How Training Actually Works (Big Picture)

Every training step follows this loop:

1. Take input tokens
2. Predict next tokens
3. Compare with actual output (loss)
4. Update the model

---

## Visual Flow

```
Input → Model → Prediction → Loss → Update → Repeat
```

---

## In Code

```python
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()
```

---

This loop runs thousands (or millions) of times.

That’s how the model learns.

---

# 9. Why Not Use the Entire Dataset at Once?

Because of attention.

Attention compares every token with every other token.

---

### Complexity

$$
O(n^2)
$$

If sequence length doubles → computation ~4x

---

That’s why we use smaller chunks (sequence length).

---

# 10. What Is Attention? (High-Level)

Attention allows each token to look at other tokens in the sequence.

---

### Example

Tokens:

```
[A, B, C, D]
```

Each token checks:

* Which tokens matter more?
* Which tokens should I focus on?

---

### Matrix View

```
        A    B    C    D
A     [ •    •    •    • ]
B     [ •    •    •    • ]
C     [ •    •    •    • ]
D     [ •    •    •    • ]
```

---

Each row = how much a token attends to others.

---

# 11. What Is Loss?

Loss measures how wrong the model is.

* High loss → bad prediction
* Low loss → good prediction

---

# 12. What Is a Gradient?

Gradient tells us:

👉 How to change the model to reduce error

---

### Intuition

“If I change this parameter slightly, does loss go up or down?”

---

# 13. Why Move Opposite to Gradient?

Gradient points uphill (increasing loss).

We want to go downhill.

$$
w_{\text{new}} = w - \eta \cdot \frac{dL}{dw}
$$

---

### Rule of Thumb

* Gradient positive → decrease weight
* Gradient negative → increase weight

---

# 14. What Does `backward()` Do?

```python
loss.backward()
```

👉 Computes gradients for all model parameters

---

### Important

* It does NOT update weights
* It only calculates gradients

---

Gradients are stored in:

```
parameter.grad
```

---

# 15. Full Training Loop

```python
for batch in dataloader:
    optimizer.zero_grad()
    logits = model(batch)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
```

---

# 16. Learning Rate and Stability

$$
w = w - \eta \cdot \text{gradient}
$$

---

### If learning rate is too high:

* Training becomes unstable
* Loss may explode
* Model may fail

---

### If too low:

* Training becomes very slow

---

# Core Mental Model

* Loss = how wrong we are
* Gradient = direction to improve
* Optimizer = how we update
* Training = repeated improvement

---

# Conceptual Map

You’ve now been exposed to:

* Tokens
* Vocabulary
* Sequence length
* Batch size
* Embeddings
* Attention
* Loss
* Gradients
* Backpropagation

---

You don’t need to fully understand everything yet.

As we build the model step by step, these concepts will become concrete.

---

Reference Code:
1. [Tensor demo](./part0/tensor_basics.py)
2. [Shape demo](./part0/shape_demo.py)
3. [embedding demo](./part0/embedding_demo.py)


👉 Next: **[Bigram Model](./PART_2_BIGRAM_MODEL.md) (your first working language model)**
