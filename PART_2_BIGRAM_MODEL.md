# Part 2 — Bigram Language Model

Here is the [code](bigram.py)
In this section, we build the simplest possible language model: a **Bigram Model**.

This model predicts the next token using only the current token.

No embeddings.
No neural networks.
No attention.

Pure statistical modeling.

---

# 1. What Is a Bigram Model?

A bigram model assumes:

[
P(t_{i+1} \mid t_i)
]

The next token depends only on the current token.

This is called **1-token memory**.

---

## Memory Definition

* 1-token memory → depends on previous 1 token
* 2-token memory (trigram) → depends on previous 2 tokens
* n-token memory → depends on previous n tokens

Bigram = memory size 1.

---

# 2. Building the Bigram Count Matrix

We construct a matrix:

[
\text{bigram_counts} \in \mathbb{R}^{V \times V}
]

Where:

* ( V = \text{vocab_size} )

Each entry:

[
\text{bigram_counts}[i, j]
]

Represents:

> How many times token j followed token i

---

## Visual Representation

If vocab_size = 5:

```
        Next Token
        0   1   2   3   4
Current
   0    2   5   0   1   3
   1    4   0   7   2   1
   2    1   3   2   6   0
   3    0   2   4   1   8
   4    5   1   0   3   2
```

Row = current token
Column = next token

---

# 3. Converting Counts to Probabilities

We normalize each row:

[
P(j \mid i) = \frac{\text{count}(i, j)}{\sum_k \text{count}(i, k)}
]

Now each row sums to 1.

Each row is a probability distribution over next tokens.

---

# 4. Text Generation Process

Generation algorithm:

1. Start with initial token
2. Look up probability row for that token
3. Sample next token
4. Repeat

This is probabilistic generation.

---

# 5. Observed Output Behavior

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

[
P(t_{i+1} \mid t_i)
]

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

[
V^2
]

Trigram size:

[
V^3
]

4-gram size:

[
V^4
]

Growth is exponential:

[
O(V^{memory})
]

---

## Example

If:

[
V = 100
]

Bigram:

[
100^2 = 10,000
]

Trigram:

[
100^3 = 1,000,000
]

4-gram:

[
100^4 = 100,000,000
]

This quickly becomes impossible for real vocab sizes (~50,000 tokens).

---

# 8. Core Limitation

Bigram models attempt to:

Memorize exact discrete combinations.

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

[
P(t_{i+1} \mid t_i)
]

Neural bigram:

[
P(t_{i+1} \mid \text{Embedding}(t_i))
]

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

* What a bigram model is
* How to construct probability tables
* Why generation looks semi-realistic
* Why long-term coherence fails
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
