# Part 3 — Neural Bigram Model (From Counts → Learning)

In Part 2, we built a model using **counts**.

Now we build a model that **learns those counts automatically**.

---

# Big Picture

We still model:

$$
P(t_{i+1} \mid t_i)
$$

But instead of counting:

👉 We learn this using parameters + training

---

# 1. From Table → Model

### Before (Bigram)

```text
Counts → Normalize → Probabilities
```

### Now (Neural Bigram)

```text
Token → Embedding → Logits → Loss → Update → Repeat
```

---

# 2. Model

```python
class NeuralBigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape

        logits = logits.view(B * T, C)
        targets = targets.view(B * T)

        loss = F.cross_entropy(logits, targets)
        return logits, loss
```

---

# 3. What Is This Embedding Layer?

```python
nn.Embedding(vocab_size, vocab_size)
```

👉 Think of it as:

```text
A table of size (V × V)
```

Each row:

```text
Token → scores for next token
```

---

# 4. What Are Logits?

Logits are:

👉 Raw scores (not probabilities)

Example:

```text
[2.0, 1.0, 0.1]
```

They mean:

* Token 0 is preferred
* Token 1 less likely
* Token 2 unlikely

---

# 5. Why We Need Softmax

Logits are not probabilities.

We need:

$$
0 \le P \le 1,\quad \sum P = 1
$$

---

## Softmax Formula

$$
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

---

## Visual Intuition

```text
Logits:        [2.0, 1.0, 0.1]

Exponentiate:  [7.39, 2.71, 1.10]

Normalize:     [0.65, 0.24, 0.11]
```

---

👉 Softmax does two things:

1. Makes values positive
2. Converts them into probabilities

---

# 6. Why We DO NOT Call Softmax Manually

We use:

```python
loss = F.cross_entropy(logits, targets)
```

NOT:

```python
probs = softmax(logits)
loss = cross_entropy(probs, targets)
```

---

## Why?

Because `F.cross_entropy` internally does:

$$
\text{log\_softmax}(logits)
$$

---

### Benefits

* More numerically stable
* Avoids overflow (`e^{1000}` problem)
* Faster and optimized

---

# 7. What Is Cross Entropy?

Cross entropy measures:

👉 “How wrong is the prediction?”

---

## Formula

$$
Loss = -\log(P_{\text{correct}})
$$

---

## Example

If correct token probability is:

* 0.9 → loss ≈ 0.1 (good)
* 0.5 → loss ≈ 0.69
* 0.01 → loss ≈ 4.6 (bad)

---

## Visual

```text
Prediction confidence vs loss:

High confidence correct  → small loss
Low confidence           → medium loss
Wrong & confident        → huge loss
```

---

👉 Training tries to:

* Increase correct probabilities
* Decrease wrong ones

---

# 8. Why We Reshape

```python
logits = logits.view(B * T, C)
targets = targets.view(B * T)
```

---

## Why?

Each token prediction is independent.

---

## Visual

```text
Before:
(B, T, C)

After:
(B*T, C)
```

---

👉 Think of it as:

```text
We solve B*T classification problems at once
```

---

# 9. Training Loop (Where Learning Happens)

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Step-by-Step

---

### 1. optimizer.zero_grad()

👉 Clears old gradients

---

### Why?

Gradients accumulate in PyTorch.

Without this:

```text
new_grad = old_grad + current_grad
```

👉 That would break training.

---

---

### 2. loss.backward()

👉 Computes gradients

---

## Math

We compute:

$$
\frac{\partial Loss}{\partial w}
$$

For every parameter.

---

### Intuition

```text
If I change this weight slightly,
how does loss change?
```

---

👉 This uses **chain rule (backpropagation)**

---

### Visual Flow

```text
Loss
 ↑
Logits
 ↑
Embedding weights
```

Gradients flow backward.

---

### 3. optimizer.step()

👉 Updates model weights using computed gradients

---

## Update Rule

$$
w = w - \eta \cdot \frac{dL}{dw}
$$

---

## What is $\eta$?

$\eta$ (eta) is the **learning rate**.

👉 It controls **how big a step we take during each update**

---

## Intuition

Think of training like walking downhill on a mountain:

- Gradient → tells you direction (which way is downhill)
- Learning rate → tells you **how big a step to take**

---

## Visual

```text
Too small step:
[ w ] → . → . → . → (very slow progress)

Good step:
[ w ] ------→ (fast and stable)

Too large step:
[ w ] -----------→ (overshoots and oscillates)
````

---

## What Happens If Learning Rate Is…

### 🔹 Too Small

* Training is very slow
* May take forever to converge
* Gets stuck easily

---

### 🔹 Too Large

* Overshoots minimum
* Loss may increase
* Training becomes unstable
* Can produce NaN values

---

## Example

If:

$$
\eta = 0.01
$$

Small updates → stable but slow

If:

$$
\eta = 1.0
$$

Very large updates → likely unstable

---

## In Code

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
```

👉 Here:

$$
\eta = 0.01
$$

---

## Why 1e-2 Here?

* Small model
* Simple task (character prediction)
* Fast experimentation

👉 Works well as a starting point

---

## Important Insight

Learning rate is one of the **most important hyperparameters**.

👉 It directly affects:

* Speed of learning
* Stability of training
* Final model quality

---

## Bonus (Good to Know)

```python
torch.optim.Adam
```

Modern optimizers like **Adam**:

* Adjust learning rate internally
* Use adaptive updates per parameter

👉 So even if global $\eta$ is fixed, effective updates vary

---

## Mental Model

```text
Gradient = direction
Learning rate = step size
Optimizer = how we walk
```

---

### Other optimizers

* SGD → simple but slower
* RMSProp → adaptive
* Adam → best default choice

---

# 🔁 Training Flow

```text
Input → Logits → Loss → Gradients → Update → Repeat
```

---

# 10. Generation (Inference)

```python
logits = logits[:, -1, :]
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

---

## Why Only Last Token?

Because:

$$
P(t_{i+1} \mid t_i)
$$

👉 Only last token matters

---

# 11. Temperature (Control Randomness)

```python
probs = F.softmax(logits / temperature, dim=-1)
```

---

## Effect

| Temperature | Behavior      |
| ----------- | ------------- |
| Low         | deterministic |
| Medium      | balanced      |
| High        | random        |

---

# 12. What This Model Learns

Same as bigram:

```text
"h" → "e"
"q" → "u"
```

---

But now:

👉 It learns instead of counting

---

# 13. Limitations

Still:

* Only 1-token context
* No structure
* No long-term memory

---

# 14. Bigram vs Neural Bigram

| Feature     | Bigram | Neural Bigram    |
| ----------- | ------ | ---------------- |
| Storage     | counts | weights          |
| Learning    | none   | gradient descent |
| Flexibility | low    | higher           |

---

# 15. Core Mental Model

```text
Token → Row Lookup → Logits → Softmax → Loss → Update
```

---

# Why This Step Matters

This is your first real neural model.

You now have:

* Parameters
* Gradients
* Optimization
* Training loop

---

👉 This is the foundation of all LLMs.

---