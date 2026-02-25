# Part 1 — Data Pipeline & Training Batch Construction

This section converts theory into concrete tensors.

> Check [code](tokenizer.py), part 0 and part 1

We move from:

Raw text → Tokens → Integer IDs → Tensor → Training Batches

This is the complete data preparation pipeline used before training a language model.

---

# 1. Loading the Dataset

We use Tiny Shakespeare as a small training corpus.

```python
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

The dataset is treated as **one continuous stream of characters**.

Important:
We do not split by sentences.
We do not split by paragraphs.
We treat the novel as raw token sequence.

---

# 2. Building the Vocabulary

We compute all unique characters:

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
```

Mathematically:

$$
\text{vocab\_size} = |\text{unique characters}|
$$

Example:
If 65 unique characters exist → vocab_size = 65

---

# 3. Creating Token Mappings

We create two dictionaries:

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

* `stoi` = string → index
* `itos` = index → string

This is the tokenizer.

---

# 4. Encoding the Dataset

We convert the entire novel into integer IDs:

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

Important:

* dtype must be `torch.long`
* Embedding layers require integer indices

Shape of data:

$$
\text{data.shape} = (N,)
$$

Where:

$$
N = \text{total number of characters}
$$

This is a 1D tensor representing the entire novel.

---

# 5. Train / Validation Split

We split dataset:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

This allows evaluation on unseen data.

---

# 6. Sliding Window Training

Language modeling objective:

Given tokens:

$$
[t_0, t_1, t_2, \ldots, t_{127}]
$$

Predict:

$$
[t_1, t_2, t_3, \ldots, t_{128}]
$$

Target is shifted by one position.

---

# 7. Batch Construction

We define:

```python
batch_size = B
sequence_length = T
```

We randomly sample starting positions:

```python
ix = torch.randint(len(data_source) - T, (B,))
```

Then construct:

```python
x = torch.stack([data_source[i:i+T] for i in ix])
y = torch.stack([data_source[i+1:i+T+1] for i in ix])
```

Resulting shapes:

$$
x.shape = (B, T)
$$

$$
y.shape = (B, T)
$$

---

# 8. Why Random Sampling?

Instead of sequential batches, we randomize because:

1. Reduces correlation between updates
2. Improves convergence
3. Ensures batches approximate full data distribution
4. Prevents early bias toward beginning of book

This is core to Stochastic Gradient Descent.

---

# 9. How Many Predictions Per Batch?

If:

$$
B = 32
$$

$$
T = 128
$$

Then model produces:

$$
32 \times 128 = 4096
$$

predictions per forward pass.

Each token position generates one prediction.

---

# 10. Data Flow So Far

```
Raw Text
   ↓
Unique Characters
   ↓
Character → Integer Mapping
   ↓
1D Tensor of Token IDs
   ↓
Random Sliding Windows
   ↓
(Batch Size, Sequence Length)
```

We now have everything required to train a language model.

---

# Key Concepts Reinforced

* Vocabulary size ≠ sequence length
* Batch size ≠ context window
* Token IDs are integers (indices)
* Embeddings are float tensors
* Targets are shifted by 1 for next-token prediction
* Random sampling improves training stability

---

# What Comes Next

Next step:

Part 2 — Build a Bigram Language Model

Before transformers, we implement the simplest possible language model to understand prediction mechanics and generation.

We move from data → probability model → text generation.

No attention yet.
No deep network yet.
Pure statistical intuition.
