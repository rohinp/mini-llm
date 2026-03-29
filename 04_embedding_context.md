# Part 4 — From Neural Bigram → Embeddings → Context → Why Transformers Exist

---

# 1. Where We Left Off

In Part 3, we built:

$$
P(\text{next token} \mid \text{current token})
$$

Model:

```text
Embedding table (V × V) → logits
```

---

## Limitation

```text
❌ Only 1-token memory
❌ No understanding of structure
❌ No long-range context
```

---

# 2. What We Actually Want

Real language depends on **context**, not just one token.

We want:

$$
P(\text{next token} \mid \text{previous tokens})
$$

Example:

```text
"I am going to the ___"
```

→ "store", "gym", "office" (depends on context)

---

👉 This requires:

* memory
* representation
* generalization

---

# 3. Step 1 — Introduce Embeddings

We move from:

```text
token → logits
```

to:

```text
token → vector → logits
```

---

## Model

```python
nn.Embedding(vocab_size, n_embd)
nn.Linear(n_embd, vocab_size)
```

---

## Shape Flow

```text
(B, T) → (B, T, n_embd) → (B, T, vocab_size)
```

---

# 4. What Is an Embedding? (Important)

An embedding is:

👉 A learned vector representation of a token

---

## Example

```text
"h" → [0.12, -0.44, 0.91, ...]
```

---

## Key Insight

```text
Similar tokens → similar vectors
```

---

## Why This Works

During training:

* tokens used in similar contexts
* get similar gradients
* end up close in vector space

---

## Visual

```text
Vector space:

      king
        ↑
 man →   → woman
        ↓
      queen
```

👉 Meaning emerges from geometry

---

# 5. What Does the Linear Layer Do?

```python
nn.Linear(n_embd, vocab_size)
```

---

## Formula

$$
y = xW + b
$$

---

## Intuition

```text
Embedding → transformed → logits
```

---

👉 It maps "meaning" → "next token scores"

---

# 6. Full Forward Flow

```text
Token IDs
   ↓
Embedding Lookup
   ↓
Dense Vector (meaning)
   ↓
Linear Layer
   ↓
Logits
```

---

# 7. Training vs Inference

## Training

```text
logits → cross_entropy → loss → backward → update
```

## Inference

```text
logits → softmax → probabilities → sampling
```

---

# 8. Important Observation (CRITICAL)

Even with embeddings:

```text
Each token is processed independently
```

---

👉 This model still learns:

$$
P(\text{next token} \mid \text{current token})
$$

---

# 🚨 This is the key limitation

---

# 9. The Context Problem

Language depends on multiple tokens:

```text
"The bank near the river"
"The bank approved the loan"
```

Same word → different meaning

---

👉 We need:

$$
P(\text{next token} \mid t_1, t_2, ..., t_n)
$$

---

# 10. How Do We Combine Tokens?

We explored 3 approaches.

---

## Option 1 — Concatenation

```text
[t1, t2, t3] → [e1 | e2 | e3]
```

---

### Visual

```text
[e1][e2][e3] → one long vector
```

---

### Pros

✔ preserves order
✔ expressive

---

### Cons

```text
❌ grows with sequence length
❌ fixed context size
❌ not scalable
```

---

## Option 2 — Average Embeddings

```text
[e1 + e2 + e3] / 3
```

---

### Visual

```text
[e1] + [e2] + [e3]
        ↓
     average vector
```

---

### Pros

✔ simple
✔ fixed size

---

### Cons

```text
❌ loses order
❌ "dog bites man" = "man bites dog"
❌ behaves like bag-of-words
```

---

## Option 3 — MLP on Concatenation

```text
concat → neural network → logits
```

---

### Pros

✔ more expressive

---

### Cons

```text
❌ still fixed size
❌ still scales poorly
```

---

# 11. Core Limitation (Important Insight)

All approaches:

```text
❌ compress multiple tokens into ONE vector
```

---

👉 Information gets lost.

---

# 12. Why This Fails (Critical Example)

```text
"dog bites man"
"man bites dog"
```

---

### Average embeddings

```text
→ same representation
```

---

👉 Order is lost → meaning destroyed

---

# 13. What We Actually Need

Instead of:

```text
many tokens → one vector
```

We want:

```text
each token → interacts with other tokens
```

---

# 14. Enter Attention (Transformer Idea)

Transformers do:

```text
each token attends to all other tokens
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
👉 No information loss
👉 Dynamic relationships

---

# 15. Why Attention Is Expensive

If sequence length = n

Each token compares with all others:

$$
O(n^2)
$$

---

## Visual

```text
n tokens → n × n interactions
```

---

## Example

```text
n = 128 → 16,384 interactions
```

---

Doubling:

```text
128 → 256 → 4x computation
```

---

# 16. Why Context Length Is Limited

Not conceptual — practical:

```text
Limited by memory and compute
```

---

👉 Model does NOT forget
👉 It simply cannot see beyond window

---

# 17. Key Mental Model

| Concept        | Meaning                |
| -------------- | ---------------------- |
| Embedding      | meaning representation |
| Linear layer   | meaning → prediction   |
| Context        | multiple tokens        |
| Attention      | token interaction      |
| Context window | max visible tokens     |

---

# Where You Are Now

You now understand:

✔ neural bigram
✔ embeddings
✔ representation learning
✔ why single-token models fail
✔ why naive context methods fail
✔ why attention is needed

---

# What Comes Next

👉 Part 5 — Attention (Core of Transformers)

We will build:

```text
Query → Key → Value → Attention weights
```

---

This is where things become powerful.
