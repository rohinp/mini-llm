# 🧠 From Neural Bigram → Embeddings → Context → Transformer Motivation

---

# 1. Where We Left Off: Neural Bigram

We built:

```text
P(next_token | current_token)
```

Model:

```text
W shape = (V, V)
```

* row = current token
* column = next token
* values = logits

---

## Limitation

```text
❌ Only 1-token memory
❌ No understanding of structure
❌ No long-range dependencies
```

---

# 2. Transition: Why Move Beyond Bigram?

We want:

```text
P(next_token | previous tokens)
```

This requires:

* richer representation
* ability to generalize
* ability to use context

---

# 3. Embeddings: First Real Representation

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

# 4. What Are Embeddings?

```text
embedding = learned vector representation of a token
```

---

## Key Insight

```text
Embeddings capture meaning via geometry
```

Tokens with similar behavior:

```text
→ similar vectors
```

---

## Where similarity is learned

```python
x = self.token_embedding_table(idx)
```

👉 NOT in the linear layer

---

# 5. Role of Each Component

| Component    | Role                 |
| ------------ | -------------------- |
| Embedding    | learn meaning        |
| Linear layer | map meaning → logits |

---

# 6. Linear Layer Explained

```python
nn.Linear(in_features, out_features)
```

Computes:

```text
y = xW + b
```

In our case:

```text
(embedding) → (logits for vocab)
```

---

# 7. Training vs Inference

## Training

```text
logits → cross_entropy → loss → backward → update weights
```

## Inference

```text
logits → softmax → probabilities → sampling
```

---

# 8. Sampling and Temperature

```text
probs = softmax(logits / temperature)
```

---

## Behavior

| Temperature | Effect        |
| ----------- | ------------- |
| low (~0)    | deterministic |
| 1.0         | normal        |
| high        | random        |

---

# 9. New Limitation (VERY IMPORTANT)

Even with embeddings:

```text
❌ each token processed independently
```

Still:

```text
P(next_token | current_token)
```

---

# 10. Introducing Context

We now want:

```text
P(next_token | last N tokens)
```

---

# 11. How to Combine Token Embeddings?

We explored 3 methods:

---

## Option 1: Concatenation

```text
[t1, t2, t3] → [e1 | e2 | e3]
```

---

### Pros

* preserves order
* expressive

---

### Cons

```text
❌ input size grows with sequence length
❌ fixed context length
❌ poor scalability
```

---

## Option 2: Average Embeddings

```text
[t1, t2, t3] → average(e1, e2, e3)
```

---

### Pros

* fixed size
* efficient

---

### Cons

```text
❌ loses order
❌ cannot capture structure
❌ bag-of-words behavior
```

---

## Option 3: MLP (on concatenation)

```text
concat → neural network → logits
```

---

### Pros

* more expressive

---

### Cons

```text
❌ still fixed input size
❌ still scales poorly
```

---

# 12. Critical Insight

All these methods:

```text
❌ combine tokens into ONE fixed vector
```

This is the core limitation.

---

# 13. Why This Fails

Example:

```text
"dog bites man"
"man bites dog"
```

Average embeddings:

```text
→ same representation
```

👉 meaning is lost

---

# 14. Enter Transformers (Motivation)

Instead of:

```text
combine → single vector
```

Transformers do:

```text
each token attends to all other tokens
```

From lecture:

```text
each word attends to each other word
```



---

# 15. Attention Complexity

```text
O(n²)
```

Because:

```text
each token interacts with every other token
```

---

## Implication

Doubling sequence length:

```text
2n → (2n)² = 4n²
```

👉 4x compute + memory

---

# 16. Why LLMs Have Context Limits

Not conceptual — practical:

```text
max_context_length = engineering constraint
```

Due to:

* memory
* compute
* attention cost

---

## Important Clarification

```text
model doesn’t "lose context"
```

👉 it simply **cannot see beyond limit**

---

# 17. Key Mental Model

| Concept        | Meaning                   |
| -------------- | ------------------------- |
| Transformer    | flexible context modeling |
| Context window | memory limit              |

---

# 🧪 Checkpoint Q&A

---

## Q1. Where is token similarity learned?

**Answer:**
In the embedding table (`nn.Embedding`), not in the linear layer.

---

## Q2. What does embedding represent?

**Answer:**
A learned vector capturing token behavior and relationships.

---

## Q3. Why add a linear layer?

**Answer:**
To map embedding → logits over vocabulary.

---

## Q4. Why not keep vocab × vocab matrix?

**Answer:**
Embeddings allow generalization and reduce parameter rigidity.

---

## Q5. What is the limitation of embedding model?

**Answer:**
No context — each token processed independently.

---

## Q6. Why does averaging fail?

**Answer:**
It loses order → becomes bag-of-words.

---

## Q7. Why does concatenation fail?

**Answer:**
Input size grows with sequence length → not scalable.

---

## Q8. What problem do all naive methods share?

**Answer:**
They compress all tokens into a single fixed vector.

---

## Q9. What does attention solve?

**Answer:**
Dynamic interaction between tokens without fixed compression.

---

## Q10. Why is attention O(n²)?

**Answer:**
Each token compares with every other token.

---

## Q11. Why do LLMs have context limits?

**Answer:**
Due to compute and memory constraints of attention.

---

## Q12. What happens when context exceeds limit?

**Answer:**
Older tokens are truncated (not visible to model).

---

## Q13. What happens when sequence length doubles?

**Answer:**
Compute and memory increase ~4x (quadratic scaling).

---

# 🚀 Where You Are Now

You understand:

* neural bigram ✅
* embeddings ✅
* representation learning ✅
* context problem ✅
* limitations of naive approaches ✅
* motivation for transformers ✅

---

### Code:
 [Embedding and language models](embedding_language_model.py)