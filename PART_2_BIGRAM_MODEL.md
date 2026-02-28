# Part 2 — Bigram Language Model (Deep Dive)

In this section, we implemented a **Bigram Language Model**. The doccument contains both:

* The statistical concept
* The PyTorch mechanics behind it

Here is the [code](bigram.py)
This model predicts the next token using only the current token.

$$
P(t_{i+1} \mid t_i)
$$

No embeddings.
No neural networks.
No attention.

Pure statistical modeling.

---

# 1. Bigram Model Definition

A bigram model assumes:

$$
P(t_{i+1} \mid t_i)
$$

Meaning:

The next token depends only on the current token.

This is called **1-token memory**.


`The Math: Conditional Probability`

In a bigram model, we simplify the probability of a whole sentence by assuming each word depends only on its immediate predecessor. This is known as a First-Order Markov Assumption.The probability of a word $w_n$ given the previous word $w_{n-1}$ is calculated as:

$$
P(w_n | w_{n-1}) = \frac{Count(w_{n-1}, w_n)}{Count(w_{n-1})}
$$

In plain English:To find the chance of "cat" following "the," you count how many times "the cat" appears in your text and divide it by the total number of times "the" appears.

---

## Memory Definition

* 1-token memory → depends on previous 1 token
* 2-token memory (trigram) → depends on previous 2 tokens
* n-token memory → depends on previous n tokens

Bigram = memory size 1.

---

# 2. Building the Bigram Count Matrix

We construct a matrix:

$$
bigram\_counts \in \mathbb{R}^{V \times V}
$$

Where:

* ( V = vocab\_size )

Each entry:

$$
bigram\_counts [i, j]
$$

Represents:

> How many times token j followed token i

---

## Example (Small Vocab = 3)

```
bigram_counts =
[
 [2, 3, 5],
 [4, 1, 5],
 [1, 1, 8]
]
```

Row = current token
Column = next token

---

# 3. Row Normalization (Counts → Probabilities)

We convert counts into probabilities using:

```python
bigram_probs = bigram_counts.float()
bigram_probs = bigram_probs / bigram_probs.sum(dim=1, keepdim=True)
```

Let’s understand this line fully.

---

## Step 1 — Row-wise Sum

```python
bigram_probs.sum(dim=1)
```

`dim=1` means:

Sum across columns → row-wise sum.

Example:

```
Row 0: 2 + 3 + 5 = 10
Row 1: 4 + 1 + 5 = 10
Row 2: 1 + 1 + 8 = 10
```

Result:

```
[10, 10, 10]
```

Shape:

```
(3,)
```

---

## Step 2 — Why `keepdim=True`?

Without `keepdim=True`, shape = `(3,)`

With `keepdim=True`, shape = `(3, 1)`

```
[
 [10],
 [10],
 [10]
]
```

This allows proper broadcasting during division.

---

## Step 3 — Broadcasting Division

We divide:

```
(3,3) / (3,1)
```

Broadcasting expands `(3,1)` to:

```
[
 [10,10,10],
 [10,10,10],
 [10,10,10]
]
```

So division becomes:

```
[
 [2/10, 3/10, 5/10],
 [4/10, 1/10, 5/10],
 [1/10, 1/10, 8/10]
]
```

Result:

```
[
 [0.2, 0.3, 0.5],
 [0.4, 0.1, 0.5],
 [0.1, 0.1, 0.8]
]
```

Now each row sums to 1.

---

## Important Rule

After normalization:

Every row must satisfy:

$$
sum\_j P(j \mid i) = 1
$$

If a row was:

```
[0, 0, 10]
```

After normalization:

```
[0, 0, 1]
```

---

## Why Convert to Float?

We use:

```python
bigram_counts.float()
```

Because:

* Probabilities require fractional values
* Integers cannot represent decimals
* Neural networks operate in floating point

Probabilities must satisfy:

$$
0 \le p \le 1
$$

So float dtype is required.

---

# 4. Sampling with torch.multinomial

Generation uses:

```python
next_token = torch.multinomial(probs, num_samples=1).item()
```

Let’s break this down.

---

## What Is `probs`?

Example:

```
probs = [0.2, 0.5, 0.3]
```

Meaning:

* 20% chance → index 0
* 50% chance → index 1
* 30% chance → index 2

---

## What Does `torch.multinomial()` Do?

It samples an index based on probability weights.

If:

```
probs = [0, 0, 1]
```

It will always return:

```
2
```

If:

```
probs = [0.2, 0.5, 0.3]
```

It randomly returns:

* 1 about 50% of the time
* 2 about 30% of the time
* 0 about 20% of the time

---

## What Does `num_samples=1` Mean?

It returns one sampled index.

If `num_samples=3`, it would return three sampled indices.

---

## Why `.item()`?

`multinomial()` returns a tensor:

```
tensor([1])
```

`.item()` converts it into a Python integer:

```
1
```

---

# 5. Why Sampling Creates Diversity

If we used:

```python
torch.argmax(probs)
```

The model would always choose the most likely token.

That would make output deterministic and repetitive.

Sampling introduces controlled randomness.

This is what gives text generation variety.

---

# 6. Observed Output Behavior

Example output:

```
Ayoowifemencofllonondsoul, ay, l his LI wde he...
```

Observations:

✔ Looks Shakespeare-like
✔ Contains realistic character transitions
✔ Preserves local structure
✘ No long-term coherence
✘ Words break apart
✘ No grammatical consistency

---

# 6. Why Does It Fail?

Because it only models:

$$
P(t_{i+1} \mid t_i)
$$

It does NOT model:

* Words
* Sentences
* Grammar
* Long-range structure

It has no memory beyond one token.

---

# 7. Scaling Problem of N-grams

If vocab_size = V

Bigram size:

$$
V^2
$$

Trigram size:

$$
V^3
$$

4-gram size:

$$
V^4
$$

Growth is exponential:

$$
O(V^{\text{memory}})
$$

---

Exponential growth:

If:

$$
V = 100
$$

Bigram:

$$
100^2 = 10{,}000
$$

Trigram:

$$
100^3 = 1{,}000{,}000
$$

4-gram:

$$
100^4 = 100{,}000{,}000
$$

This quickly becomes impossible for real vocab sizes (~50,000 tokens).

---

# 8. Core Limitation

Bigram models attempt to:

Memorize exact token transitions.

They do not learn continuous representations.

Memory requirements explode exponentially as context increases.

---

# 9. Why This Matters

The failure of n-grams motivated neural language models.

Instead of storing exact combinations:

Neural models:

* Learn embeddings
* Learn continuous representations
* Share statistical strength across tokens
* Scale more efficiently

This leads to transformers.

---

# 10. Conceptual Bridge to Neural Bigram

Statistical bigram:

$$
P(t_{i+1} \mid t_i)
$$

Neural bigram:

$$
P(t_{i+1} \mid \text{Embedding}(t_i))
$$

Instead of storing counts:

We learn probability distributions via parameters.

This introduces:

* Embedding layer
* Linear layer
* Softmax
* Cross-entropy loss
* Backpropagation

---

# Summary of Part 2

You now understand:

* How bigram probability tables are constructed
* How row normalization works (mechanically)
* Why `keepdim=True` matters
* How broadcasting works in PyTorch
* Why probabilities require floats
* How `torch.multinomial()` samples
* Why sampling creates creative output
* Why n-grams scale exponentially
* Why deep learning replaced discrete n-grams

---

# What Comes Next

Part 3 — Neural Bigram Model

We replace the count table with:

Embedding → Linear Layer → Softmax

This introduces:

* Learnable parameters
* Gradient descent
* Real training loop

And bridges us toward transformers.
