# 🧠 Part 5 — Context-Aware Model (Concatenation Approach)

---

# 1. Where We Are Coming From

So far, we have built models like:

$$
P(\text{next token} \mid \text{current token})
$$

Even with embeddings, we were still doing:

```text
token → embedding → logits
```

---

## 🚨 Problem

```text
❌ Only 1-token context
❌ Cannot capture patterns like "New York"
❌ Cannot understand structure
```

---

# 2. New Goal

We now want:

$$
P(\text{next token} \mid t_1, t_2, ..., t_N)
$$

Where:

```text
N = block_size (context length)
```

---

## Example

```text
Input:  "I am going to"
Output: " school"
```

👉 Prediction depends on **multiple tokens**

---

# 3. Core Idea

Instead of treating tokens independently:

```text
[t1, t2, t3, ..., tN]
   ↓
[e1, e2, e3, ..., eN]
   ↓
concatenate → one big vector
   ↓
linear layer → logits
```

---

## Visual

```text
e1   e2   e3   e4
↓    ↓    ↓    ↓
[---concatenate---]
          ↓
   one long vector
          ↓
      prediction
```

---

# 4. Model Intuition

👉 We are saying:

> “Take all token meanings, glue them together, and predict the next token.”

---

# 5. Hyperparameters

```python
batch_size = 32
block_size = 8     # how many tokens we look at
n_embd = 32        # size of each embedding vector
learning_rate = 1e-3
```

---

## What They Control

| Parameter  | Meaning              |
| ---------- | -------------------- |
| block_size | context length       |
| n_embd     | representation power |
| batch_size | parallel training    |

---

# 6. Batch Construction (Important Shift)

```python
x = torch.stack([data[i:i+block_size] for i in ix])  # (B, block_size)
y = torch.stack([data[i+block_size] for i in ix])    # (B)
```

---

## What This Means

```text
Input:  [t1 t2 t3 t4]
Target: t5
```

---

## Key Difference from Before

Before:

```text
(B, T) → predict T tokens
```

Now:

```text
(B, block_size) → predict 1 token
```

---

👉 The model now uses **context to predict one next token**

---

# 7. Model Implementation

```python
class ContextConcatModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.block_size = block_size

        self.lm_head = nn.Linear(n_embd * block_size, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)  # (B, block_size, n_embd)

        B, T, C = x.shape
        x = x.view(B, T * C)  # flatten

        logits = self.lm_head(x)  # (B, vocab_size)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits, targets)
        return logits, loss
```

---

# 8. Shape Flow (Critical to Understand)

```text
Input:
(B, block_size)

Embedding:
(B, block_size, n_embd)

Flatten:
(B, block_size * n_embd)

Output:
(B, vocab_size)
```

---

## Why Flatten?

Because:

```text
Linear layer expects a fixed-size vector
```

---

# 9. Math Behind It

### Concatenation

If:

* embedding size = $d$
* context length = $T$

Then:

$$
\text{input dimension} = T \times d
$$

---

### Linear Layer

$$
y = xW + b
$$

Where:

* $x \in \mathbb{R}^{T \cdot d}$
* $W \in \mathbb{R}^{(T \cdot d) \times V}$

---

👉 This directly maps context → logits

---

# 10. Generation (Handling Fixed Input Size)

### Problem

Model expects:

```text
block_size tokens
```

But generation starts with:

```text
[1 token]
```

---

### Solution: Padding

```python
if idx_cond.shape[1] < self.block_size:
    pad = torch.zeros(...)
    idx_cond = torch.cat((pad, idx_cond), dim=1)
```

---

## Why Padding Works

```text
Padding token → learned embedding
```

---

👉 Important:

```text
Embedding(0) is NOT empty
```

It is:

* a real vector
* learned during training
* ignored over time

---

# 11. What This Model Learns

Compared to previous models:

```text
✔ learns short phrases
✔ uses context
✔ better predictions
```

---

# 12. Core Limitation (Very Important)

We are doing:

```text
multiple tokens → ONE vector
```

---

## Why This Is Bad

```text
Information gets compressed
```

---

## Example

```text
"dog bites man"
"man bites dog"
```

After averaging or compression:

```text
→ similar representation
```

👉 meaning is lost

---

# 13. Scaling Problem

Input dimension:

$$
T \times d
$$

---

### Example

| block_size | input_dim |
| ---------- | --------- |
| 8          | 256       |
| 32         | 1024      |
| 128        | 4096      |

---

👉 This grows quickly → inefficient

---

# 14. Bigger Problem (Conceptual)

All tokens are:

```text
forced into ONE fixed vector
```

---

👉 This creates a bottleneck

---

# 15. What We Actually Need

Instead of:

```text
combine → one vector
```

We want:

```text
each token interacts with others
```

---

# 16. Transition to Attention

```text
t1 ↔ t2 ↔ t3 ↔ t4
```

---

## Visual

```text
t1 → t2, t3, t4
t2 → t1, t3, t4
t3 → t1, t2, t4
```

---

👉 No compression
👉 Dynamic relationships

---

# 17. Why This Matters

We now understand:

```text
❌ fixed vector = limitation
✔ dynamic interaction = solution
```

---

# 18. Summary Comparison

| Model         | Context  | Representation | Limitation   |
| ------------- | -------- | -------------- | ------------ |
| Bigram        | 1 token  | lookup table   | no context   |
| Neural Bigram | 1 token  | learned        | no context   |
| Concat Model  | N tokens | fixed vector   | not scalable |

---

# 🚀 Where You Are Now

You now understand:

✔ embeddings
✔ context modeling
✔ sequence inputs
✔ why naive approaches fail
✔ why scaling breaks

---

# 🎯 Next Step

👉 **Self-Attention (Transformer Core)**

We will move from:

```text
fixed combination
```

to:

```text
dynamic token interaction
```

---

This is where modern LLMs begin.
