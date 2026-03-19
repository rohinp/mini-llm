# 🧠 Neural Bigram Language Model — From Scratch

## 🎯 Goal

Understand how a **neural bigram language model** works end-to-end:

* From tokenization
* To model definition
* To training (forward + backward)
* To text generation

---

# 1. What is a Language Model?

A language model learns:

```
P(next_token | previous_tokens)
```

For a **bigram model**:

```
P(next_token | current_token)
```

👉 Only depends on the **last token**.

---

# 2. Bigram Model (Intuition)

Example:

```
Input: "h"
Output probabilities:
    "e" → 0.8
    "a" → 0.1
    "z" → 0.01
```

This is learned from data.

---

# 3. Count-Based vs Neural Bigram

### Count-based

* Uses frequency counts
* Static probabilities

### Neural Bigram

* Uses trainable weights
* Learns probabilities via optimization

---

# 4. Vocabulary and Tokenization

### Build vocabulary

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
```

### Create mappings

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

### Encode / Decode

```python
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])
```

---

# 5. Dataset Preparation

Convert entire text:

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

Split:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

---

# 6. Batch Generation

Set a couple of knobs that control how much data we look at per step:

```python
batch_size = 32       # how many chunks per gradient step
sequence_length = 8   # how many characters per chunk
```

```python
def get_batch(split, batch_size_override=None, sequence_length_override=None):
    data_source = train_data if split == "train" else val_data

    bs = batch_size if batch_size_override is None else batch_size_override
    sl = sequence_length if sequence_length_override is None else sequence_length_override

    ix = torch.randint(len(data_source) - sl, (bs,))

    x = torch.stack([data_source[i:i+sl] for i in ix])
    y = torch.stack([data_source[i+1:i+sl+1] for i in ix])

    return x, y
```

👉 Those optional override arguments let other scripts (like `neural_bigram.py`) plug in different batch sizes without editing this helper.

### Example:

```
x: "hello wo"
y: "ello wor"
```

👉 Each token predicts the **next token**

---

# 7. Neural Bigram Model

## Key Idea

We learn a matrix:

```
W shape = (vocab_size, vocab_size)
```

Where:

```
W[i][j] = logit score for next_token=j given current_token=i
```

---

# 8. Model Implementation

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

---

# 9. Important Concepts

## 🔹 Embedding Layer

```python
nn.Embedding(vocab_size, vocab_size)
```

Acts as:

```
token_id → row lookup → logits vector
```

Equivalent to:

```
one_hot @ W
```

---

## 🔹 One-Hot Vectors vs Embeddings

*A one-hot vector* is a long vector full of zeros except for a single 1 at the position of the token. If the vocabulary has 65 tokens, each one-hot vector is length 65.

*The embedding layer* stores all of those one-hot → weight translations implicitly. When you call `nn.Embedding`, PyTorch takes the token ID, jumps directly to the matching row, and skips creating the sparse one-hot vector entirely.

Why this matters:

- No gigantic sparse vectors in memory.
- Updating a single row during training is trivial.
- The math matches “one_hot @ W”, but the code stays fast and simple.

---

## 🔹 Logits

Raw scores (not probabilities):

```
[2.1, 0.3, -1.7]
```

---

## 🔹 Softmax

Converts logits → probabilities:

```
sum = 1
```

---

## 🔹 Cross Entropy

Measures:

```
How wrong is the prediction?
```

---

## 🔹 Mini Number Walkthrough

Assume a 3-token vocabulary. The model returns logits:

```python
logits = torch.tensor([2.0, 0.5, -1.0])
```

Softmax turns that into probabilities:

```
[0.71, 0.21, 0.08]
```

If the true next token is index 1, the loss is `-log(0.21) ≈ 1.56`. Higher probability for the correct token → lower loss.

---

# 10. Shape Flow

Input:

```
(B, T)
```

After embedding:

```
(B, T, C)
```

After reshape:

```
(B*T, C)
```

Targets:

```
(B*T)
```

Plain-English version:

- `B` (batch size): how many independent snippets we process in parallel (e.g., 32).
- `T` (time steps / sequence length): how many characters are in each snippet (e.g., 8).
- `C` (channels / vocab size): how many possible tokens we predict (equal to vocabulary length).

During training we predict `B × T` next tokens at once. PyTorch’s `cross_entropy` expects shape `(N, C)` for logits, so we flatten `(B, T, C)` → `(B*T, C)` and flatten the targets to match. Think of it as stacking every position across the batch into one big classification table.

---

# 11. Training Loop (Backward Pass)

```python
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(2000):

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 200 == 0:
        print(step, loss.item())
```

To check whether we are improving on both train and validation splits, evaluate in `torch.no_grad()` mode every few steps:

```python
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(20)
        for k in range(20):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
```

### Why bother with `@torch.no_grad()`?

- **Without it:** PyTorch tracks gradients for every tensor operation, even during evaluation. That wastes memory and slows things down.
- **With it:** Autograd tracking is disabled inside the decorated function (or context manager), so inference/validation runs cheaper and cannot accidentally call `loss.backward()`.
- **When to use it:** whenever you are just *measuring* or *sampling* (validation loops, text generation). Training steps still run outside this context so gradients get recorded normally.

---

## 🔁 Training Flow

```
forward → loss → backward → update weights
```

---

# 12. Generation (Inference)

```python
def generate(self, idx, max_new_tokens):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            logits = logits[:, -1, :]  # last token

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

`torch.multinomial(probs, num_samples=1)` draws one token ID using the probability distribution we just computed, so more likely tokens get sampled more often but everything still has a chance.

---

## Usage

```python
context = torch.zeros((1,1), dtype=torch.long)
generated = model.generate(context, 200)

print(decode(generated[0].tolist()))
```

Remember to call `model.eval()` before sampling so any layers such as dropout switch to inference mode (our simple bigram model does not have them yet, but building the habit now helps later).

---

# 13. Training vs Inference

| Phase     | Flow                          |
| --------- | ----------------------------- |
| Training  | logits → cross_entropy → loss |
| Inference | logits → softmax → sampling   |

---

# 14. Temperature (Sampling Control)

During generation we can scale logits before softmax:

```python
probs = F.softmax(logits / temperature, dim=-1)
```

Think of temperature as a *confidence dial*:

- Low temperature → sharp, deterministic distributions.
- High temperature → flat, exploratory distributions.

### Case 1 — Low temperature (e.g., 0.5)

```
logits / 0.5  ⇒  values double
```

Example:

```
original logits : [2.0, 1.0, 0.1]
after /0.5      : [4.0, 2.0, 0.2]
softmax probs   : [0.85, 0.13, 0.02]
```

The largest value dominates, so the model becomes confident and repetitive—almost like always taking `argmax`.

### Case 2 — High temperature (e.g., 2.0)

```
logits / 2.0  ⇒  values shrink
```

Example:

```
[2.0, 1.0, 0.1] / 2 → [1.0, 0.5, 0.05]
softmax probs       → [0.50, 0.30, 0.20]
```

Probabilities spread out, so the model samples more diverse (and sometimes nonsensical) continuations.

### Quick reference

| Temperature | Behavior                    |
| ----------- | --------------------------- |
| 0.1         | almost argmax (very strict) |
| 0.5         | confident                   |
| 1.0         | baseline / neutral          |
| 2.0         | exploratory                 |
| ≫ 2         | nearly random               |

### Why it works

Softmax uses exponentials: `exp(logit)`. Larger logits explode the exponent, so one option dominates; smaller logits keep the exponentials close, so the distribution flattens. Temperature simply rescales logits before those exponentials are applied.

### Tie-in to the bigram model

Our neural bigram’s forward pass yields `logits = W[token_id]`. Right before sampling we can control creativity:

```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
```

This lets you demo the same trained weights under both “serious” (low temp) and “fun” (high temp) settings without retraining.

---

# 15. Key Insight

Bigram model learns:

```
P(next_token | current_token)
```

But cannot capture:

* long context
* grammar
* structure

👉 This leads to **transformers later**

---

# 🧪 Checkpoint Q&A

## Q1. Why don’t we store one-hot vectors?

**Answer:**
Because indexing into embedding is equivalent and much more efficient.

---

## Q2. Why is matrix shape (V, V)?

**Answer:**
Each row = current token
Each column = next token

---

## Q3. What does W[i][j] represent?

**Answer:**
Score (logit) for next_token=j given current_token=i

---

## Q4. Why reshape (B,T,C) → (B*T,C)?

**Answer:**
Each token prediction is an independent classification problem.

---

## Q5. What does embedding return?

**Answer:**
A vector of logits for all possible next tokens.

---

## Q6. Why no softmax in forward?

**Answer:**
`cross_entropy` applies it internally for numerical stability.

---

## Q7. Why softmax during generation?

**Answer:**
To convert logits into probabilities for sampling.

---

## Q8. What happens with argmax sampling?

**Answer:**

* deterministic output
* repetitive text
* no creativity

---

## Q9. What happens when temperature → 0?

**Answer:**

* behaves like argmax
* highly deterministic

---

## Q10. What happens when temperature is very high?

**Answer:**

* distribution becomes uniform
* output becomes random/nonsensical

---

## Q11. Why only use last token during generation?

**Answer:**
Bigram model only depends on current token, so only last token matters.

---

## Q12. Why isn’t backward inside the model?

**Answer:**
PyTorch autograd handles backward; model only defines forward computation.

---

# 16. Putting It All Together

1. **Vocabulary & data:** read the text, build `stoi/itos`, encode everything into a long tensor, and split into train/val.
2. **Batches:** choose `batch_size`/`sequence_length`, then call `get_batch` to fetch parallel examples `(x, y)` where each `x` predicts the next token in `y`.
3. **Model:** the embedding layer stores a learnable row of logits for every token; calling `forward` on a batch returns all logits and, if targets are provided, the cross-entropy loss.
4. **Training loop:** repeatedly `forward → loss → backward → optimizer.step()`, and every few iterations run `estimate_loss()` to monitor both splits.
5. **Generation:** switch to eval/no-grad mode, feed a starting context (e.g., `[0]`), and keep sampling the next token with `softmax + torch.multinomial`.
6. **Decode:** turn the generated token IDs back into text with `decode`.

---

### Related Code:

1. [BigramModel](bigram.py)
2. [NeuralBigramModel](neural_bigram.py)
3. [Batch and tokenizer code](tokenizer.py)
