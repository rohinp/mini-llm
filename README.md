# Build Your Own Mini-LLM

A practical, developer-first repo for understanding how language models work by building them step by step.

This project is designed for **developers and aspiring AI engineers** who want to move from:

```text
raw text → tokens → batches → bigram models → neural models → embeddings → context → transformers
```

The goal is **clarity over hype**.

You do not need to be an ML expert to start.
We take small steps, keep the math approachable, and connect every concept back to code.

---

## Who This Repo Is For

This repo is for you if:

* you are a developer curious about how LLMs work internally
* you want to build intuition, not just use APIs
* you prefer code-first learning over heavy theory
* you want to understand training, generation, embeddings, and transformers in a practical way

---

## How to Use This Repo

Think of this repo as the **table of contents of a hands-on book**.

Some concepts may not click on the first read — that is normal.
A lot of the ideas are revisited across parts, with more context each time.

👉 You do **not** need to read everything linearly if something feels too abstract.

---

## Recommended Learning Path

### 1. [Foundations](./FOUNDATIONS.md)

Start here for the mental models behind language models, tokens, embeddings, batches, gradients, and training.

This section gives you the vocabulary needed for everything that follows.

---

### 2. [Data Pipeline](./PART_1_DATA_PIPELINE.md)

Learn how raw text becomes tensors and training batches.

This part covers:

* vocabulary creation
* token-to-id mapping
* train/validation split
* batch construction
* next-token training setup

If this feels slightly abstract at first, keep going — it becomes much clearer once you see the first model.

---

### 3. [Bigram Model](./PART_2_BIGRAM_MODEL.md)

This is the best starting point for most developers.

You build the simplest possible language model:

* no neural network
* no backpropagation
* no attention

Just counts, probabilities, and text generation.

---

### 4. [Neural Bigram](./PART_3_NEURAL_BIGRAM.md)

This is where real learning begins.

You move from:

```text
counting transitions
```

to:

```text
learning transitions with parameters
```

This introduces:

* embeddings as lookup tables
* logits
* softmax
* cross-entropy loss
* backpropagation
* optimizer updates

---

### 5. [Embedding + Context Motivation](./PART_4_EMBEDDING_CONTEXT.md)

This section explains why single-token models are not enough.

You will learn:

* what embeddings really buy us
* why context matters
* why naive ways of combining context break down
* why transformers become necessary

This step is intentionally included to make the transition smoother.

---

### 6. [Context Concat Model](./PART_5_CONTEXT_CONCAT_MODEL.md)

A step toward using multiple previous tokens together.

This helps build intuition for:

* fixed context windows
* concatenation-based context modeling
* why more context helps
* why this still does not fully solve the problem

---

### 7. More Coming Next

Planned topics include:

* attention
* transformer blocks
* training improvements
* generation tricks
* running small models locally

---

## Suggested Start Point

If you are unsure where to begin:

👉 Start with [Bigram Model](./PART_2_BIGRAM_MODEL.md)

Then jump back to:

* [Foundations](./FOUNDATIONS.md)
* [Data Pipeline](./PART_1_DATA_PIPELINE.md)

That path is often easier for developers than starting with theory first.

---

## Project Philosophy

* **developer-first**
* **small steps**
* **minimal black boxes**
* **practical over academic**
* **understand what the model is doing, not just how to run it**

---

## Disclaimer

AI assistance was used to help polish documentation and support code writing where useful.

If you spot incorrect explanations, code issues, or places that need improvement, please open an issue.
