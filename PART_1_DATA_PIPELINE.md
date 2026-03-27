# Part 1 — Data Pipeline & Training Batch Construction

This section turns theory into actual tensors you can feed into a model.

👉 We go from raw text → numbers → structured batches ready for training.

> Check code: [Tokenizer code](./tokenizer.py), [tensor introduction](./part0/tensor_basics.py), [About shapes](./part0/shape_demo.py), [About embeddings](./part0/embedding_demo.py)

---

# Big Picture

We transform data step by step:

```
Raw Text → Tokens → Integer IDs → Tensor → Training Batches
```

Everything in a language model starts from this pipeline.

---

# 1. Loading the Dataset

We use Tiny Shakespeare as a small training corpus.

```python
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

---

### Important Design Choice

We treat the dataset as:

👉 **One continuous stream of characters**

We do NOT:

* split sentences
* split paragraphs
* reset context

Why?

Because language models learn patterns across arbitrary boundaries.

---

# 2. Building the Vocabulary

We extract all unique characters:

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
```

Mathematically:

$$
\text{vocab\_size} = |\text{unique characters}|
$$

---

### Developer Intuition

👉 Vocabulary = all possible symbols the model can output

---

# 3. Creating Token Mappings

We create mappings:

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

* `stoi` → string → integer
* `itos` → integer → string

---

### Why This Exists

Models cannot work with text directly.

👉 Everything must become numbers.

---

# 4. Encoding the Dataset

We convert text into integers:

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

---

### Important

* `dtype=torch.long` is required
* Embedding layers expect integer indices

---

### Shape

$$
\text{data.shape} = (N,)
$$

Where:

$$
N = \text{total number of characters}
$$

---

👉 This is a **1D tensor representing the entire dataset**

---

# 5. Train / Validation Split

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

---

### Why This Matters

* Train → model learns
* Validation → model is evaluated

👉 Prevents overfitting

---

# 6. Sliding Window Training Objective

Given:

$$
[t_0, t_1, t_2, \ldots, t_{127}]
$$

We want:

$$
[t_1, t_2, t_3, \ldots, t_{128}]
$$

---

### Key Idea

👉 Predict the **next token at every position**

This is the core of language modeling.

---

# 7. Batch Construction (Core Step)

We define:

```python
batch_size = B
sequence_length = T
```

---

### Step 1: Sample Random Start Positions

```python
ix = torch.randint(len(data_source) - T, (B,))
```

---

### Step 2: Build Input and Target

```python
x = torch.stack([data_source[i:i+T] for i in ix])
y = torch.stack([data_source[i+1:i+T+1] for i in ix])
```

---

### Shapes

$$
x.shape = (B, T)
$$

$$
y.shape = (B, T)
$$

---

### What This Means

* `x` = input tokens
* `y` = expected next tokens

👉 Same sequence, shifted by one

---

# 8. Why Random Sampling?

We don’t iterate sequentially.

We randomize because:

1. Reduces correlation between batches
2. Improves convergence
3. Better approximates full dataset
4. Avoids bias toward early text

---

👉 This is the “stochastic” part of SGD (Stochastic Gradient Descent):

We randomly sample data for each batch instead of using the entire dataset,
which makes training faster and more efficient.

Visual: Full Batch vs Stochastic Training

Without Stochastic (Full Batch Gradient Descent)
```
Entire Dataset
[ t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 ... ]

        ↓ (use ALL data)

Compute Loss → Compute Gradient → Update Weights
        ↓
   (1 slow update)
```

With Stochastic (Mini-Batch / SGD)
```
Entire Dataset
[ t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 ... ]

   ↓ random sample
[ t3 t4 t5 t6 ]

   ↓ random sample
[ t8 t9 t0 t1 ]

   ↓ random sample
[ t2 t3 t4 t5 ]

Each batch:
Compute Loss → Gradient → Update

→ Many small, fast updates
```

👉 Instead of waiting to process the whole dataset,
we learn continuously from small random pieces.

```
This is the “stochastic” part of Stochastic Gradient Descent (SGD):

- Each training step uses a randomly sampled batch
- The gradient is an approximation of the true gradient
- Updates are noisier, but much faster
- Over many steps, the model still converges

This is why training large models is even possible.

👉 Think of it like learning from random pages of a book instead of reading the whole book before every improvement.
```

---

# 9. How Much Work Happens Per Batch?

If:

$$
B = 32
$$

$$
T = 128
$$

Then:

$$
32 \times 128 = 4096
$$

predictions per forward pass.

---

👉 Every token position produces one prediction.

---

# 10. Data Flow So Far

```
Raw Text
   ↓
Unique Characters
   ↓
Character → Integer Mapping
   ↓
1D Tensor (Token IDs)
   ↓
Random Sampling
   ↓
(Batch Size, Sequence Length)
```

---

# 11. Alternative Batch Strategy (Block Prediction)

So far, each input predicts a **sequence of next tokens**.

Now let’s look at a slightly different approach.

---

## Code

```python
def get_batch_with_block(split, block_size, batch_size_override=None):
    data = train_data if split == "train" else val_data
    bs = batch_size if batch_size_override is None else batch_size_override
    ix = torch.randint(len(data) - block_size, (bs,))

    x = torch.stack([data[i : i + block_size] for i in ix])  # (B, block_size)
    y = torch.stack([data[i + block_size] for i in ix])      # (B)

    return x, y
```

---

## What’s Different?

### Input

```python
x.shape = (B, block_size)
```

A sequence of tokens.

---

### Target

```python
y.shape = (B,)
```

👉 Only ONE token per sequence.

---

## Intuition

Instead of predicting at every step:

👉 We predict **only the next token after the block**

---

### Example

If block_size = 4:

```
Input (x):  [t0, t1, t2, t3]
Target (y): t4
```

---

## Why Use This?

This setup is useful when:

* You only care about the **final prediction**
* You want simpler outputs
* You’re experimenting with simpler models (like early bigram/MLP setups)

---

## Comparison

| Approach         | Input Shape | Target Shape | Predictions    |
| ---------------- | ----------- | ------------ | -------------- |
| Sliding Window   | (B, T)      | (B, T)       | T per sequence |
| Block Prediction | (B, T)      | (B,)         | 1 per sequence |

---

👉 Sliding window = full supervision
👉 Block prediction = single-step supervision

---

# Key Concepts Reinforced

* Vocabulary size ≠ sequence length
* Batch size ≠ context window
* Token IDs are integers
* Embeddings are float vectors
* Targets are shifted for next-token prediction
* Random sampling improves training

---

# What Comes Next

Next step:

👉 **Part 2 — Bigram Language Model**

We will:

* Use this data pipeline
* Build the simplest possible model
* Generate real text

No attention yet.
No deep networks yet.

Just pure mechanics of prediction.
