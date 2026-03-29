# Part 2 — Bigram Language Model (Deep Dive)

In this section, we build the **simplest possible language model**.

👉 No neural networks
👉 No embeddings
👉 No training loop

Just counting patterns in data.

---

## Big Picture

We want to answer:

👉 “Given a token, what is the most likely next token?”

Mathematically:

$$
P(t_{i+1} \mid t_i)
$$

This is called a **Bigram Model**.

---

# 1. What Is a Bigram Model?

A bigram model assumes:

👉 The next token depends **only on the current token**

---

### Example

If current token = `'t'`
Model predicts:

* `'h'` with high probability
* `'x'` with low probability

---

### Memory Definition

* Bigram → looks at **1 previous token**
* Trigram → looks at **2 tokens**
* N-gram → looks at **N tokens**

👉 Bigram = **1-token memory**

---

# 2. From Text → Counts

We scan the dataset and count transitions:

```python
for i in range(len(data) - 1):
    current_token = data[i].item()
    next_token = data[i + 1].item()
    bigram_counts[current_token, next_token] += 1
```

---

## What This Builds

A matrix:

$$
\text{bigram\_counts} \in \mathbb{R}^{V \times V}
$$

Where:

* Row = current token
* Column = next token

---

### Intuition

```text
Row = "what I have now"
Column = "what comes next"
```

---

## Example

```id="example_matrix"
[
 [2, 3, 5],
 [4, 1, 5],
 [1, 1, 8]
]
```

👉 If current token = row 0
→ next token probabilities come from that row

---

# 3. Counts → Probabilities

Raw counts are not enough—we need probabilities.

---

## Code

```python
bigram_probs = bigram_counts.float()
bigram_probs = bigram_probs / bigram_probs.sum(dim=1, keepdim=True)
```

---

## What’s Happening?

### Step 1 — Convert to Float

Counts are integers:

```
2, 3, 5
```

We convert to:

```
2.0, 3.0, 5.0
```

👉 Because probabilities need decimals.

---

### Step 2 — Normalize Rows

Each row becomes a probability distribution.

---

### Example

Before:

```
[2, 3, 5]
```

Sum:

```
10
```

After:

```
[0.2, 0.3, 0.5]
```

---

## Important Rule

Every row must sum to:

$$
1
$$

Because it represents:

👉 “All possible next tokens”

---

## Why `keepdim=True`?

Without it:

* shape = `(V,)`

With it:

* shape = `(V, 1)`

👉 This allows proper broadcasting during division.

---

# 4. Generation (The Fun Part)

We now generate text using probabilities.

---

## Code

```python
probs = bigram_probs[current]
next_token = torch.multinomial(probs, num_samples=1).item()
```

---

## What Is Happening?

We sample the next token based on probabilities.

---

### Example

```
probs = [0.2, 0.5, 0.3]
```

* 50% → token 1
* 30% → token 2
* 20% → token 0

---

👉 This is **probabilistic selection**, not deterministic.

---

# 5. Why Not Use Argmax?

If we use:

```python
torch.argmax(probs)
```

We always pick the highest value.

👉 Output becomes repetitive and boring.

---

### Sampling gives:

* variation
* creativity
* more realistic text

---

# 6. What the Model Actually Learns

This model learns:

👉 “What characters follow other characters”

---

### Example Behavior

```
"q" → almost always followed by "u"
```

---

# 7. Output Behavior

Example:

```
Ayoowifemencofllonondsoul, ay, l his LI wde he...
```

---

### What It Gets Right

✔ Local patterns
✔ Character transitions
✔ Looks vaguely like Shakespeare

---

### What It Gets Wrong

✘ No words
✘ No grammar
✘ No long-term structure

---

# 8. Why It Fails

Because it only models:

$$
P(t_{i+1} \mid t_i)
$$

It has:

* No memory beyond 1 token
* No understanding of words
* No structure

---

# 9. Scaling Problem (Why This Doesn’t Work)

If vocabulary size = $V$

---

### Bigram

$$
V^2
$$

### Trigram

$$
V^3
$$

---

### Example

If:

$$
V = 100
$$

Then:

* Bigram → 10,000
* Trigram → 1,000,000
* 4-gram → 100,000,000

---

👉 Growth is exponential:

$$
O(V^{\text{memory}})
$$

---

# 10. Core Limitation

Bigram models:

* Memorize exact transitions
* Do NOT generalize
* Do NOT learn representations

---

# 11. Why Neural Models Replaced This

Instead of storing counts:

Neural models:

* Learn embeddings
* Share patterns across tokens
* Scale efficiently

---

# 12. Bridge to Neural Bigram

Statistical:

$$
P(t_{i+1} \mid t_i)
$$

Neural:

$$
P(t_{i+1} \mid \text{Embedding}(t_i))
$$

---

👉 Instead of lookup tables, we learn parameters.

---

# 13. Code Summary (Connect Everything)

```python
# Build counts
bigram_counts = torch.zeros((vocab_size, vocab_size))
...

# Convert to probabilities
bigram_probs = bigram_counts.float()
bigram_probs = bigram_probs / bigram_probs.sum(dim=1, keepdim=True)

# Generate text
next_token = torch.multinomial(probs, num_samples=1).item()
```

---

# Core Mental Model

```id="mental_model"
Current Token → Look Row → Get Probabilities → Sample → Next Token
```

Repeat this loop → generate text.

---

# Summary

You now understand:

* How bigram tables are built
* How probabilities are computed
* How sampling works
* Why output looks “kind of real”
* Why it fails for real language

---

# What Comes Next

👉 Part 3 — Neural Bigram Model

We replace counts with:

Embedding → Softmax

Now the model will:

* Learn instead of memorize
* Use gradients
* Train like real neural networks

---
