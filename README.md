# Attention From Scratch (Transformer Core)

This project is a from-scratch implementation of Self-Attention and Multi-Head Self-Attention, inspired by the paper “Attention Is All You Need” (Vaswani et al., 2017).

The goal of this project is to understand and implement the core mechanism behind modern language models such as ChatGPT, BERT, and other Transformer-based systems — without training large models or using datasets.

###  What This Project Covers
This repository implements:
* ✅ **Single-Head Self-Attention**
* ✅ **Multi-Head Self-Attention**
* ✅ **Scaled Dot-Product Attention**
* ✅ **Query, Key, Value (Q, K, V) mechanism**
* ✅ **Clean, readable PyTorch implementation**
* ❌ **No training**
* ❌ **No datasets**
* ❌ **No backpropagation**

This keeps the focus purely on understanding how attention works.

---

###  Why Attention?
Before Transformers, sequence models like RNNs and LSTMs processed text one word at a time, which made them:
* **slow**
* **hard to parallelize**
* **weak at long-range dependencies**

The Transformer introduced self-attention, allowing every word to look at every other word at once, making models faster, more scalable, and more powerful. This project implements that exact idea.

---

###  Core Idea: Self-Attention
For a given input sentence represented as embeddings:
Each word is projected into:
* **Query (Q)** – what the word is looking for
* **Key (K)** – what the word offers
* **Value (V)** – the information the word carries

Attention scores are computed using:
$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

These scores are used to mix values (V), producing context-aware word representations.

---

### Single-Head Self-Attention
The `SelfAttention` class implements one attention head, which demonstrates the core mechanism from the paper.

**Input** `(batch_size, sequence_length, embedding_dim)`
**Output** `(batch_size, sequence_length, embedding_dim)`

This version is useful for:
1. understanding the fundamentals
2. seeing how Q, K, V interact
3. building intuition before scaling

---

###  Multi-Head Self-Attention
The `MultiHeadSelfAttention` class extends the single-head version. Instead of one attention mechanism, it uses multiple attention heads in parallel, where each head:
1. attends to the sentence independently
2. focuses on different representation subspaces
3. The outputs of all heads are: **concatenated** and **projected** back to the original embedding size

#### Why Multi-Head Attention?
Multi-head attention allows the model to simultaneously capture:
* **grammatical relationships**
* **semantic meaning**
* **positional patterns**
* **long-range dependencies**

This is one of the key reasons Transformers scale so well.

---

###  Project Structure
```text
attention-from-scratch/
│
├── attention.py       # Self-Attention and Multi-Head Self-Attention implementations
├── demo.py            # Simple demo showing how both modules work
├── README.md          # Project explanation
└── requirements.txt   # Dependencies

```
---

▶️ How to Run:

1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the demo
python demo.py

Example Output

=== Single-Head Self-Attention ===
Output shape: torch.Size([1, 3, 4])
Attention weights shape: torch.Size([1, 3, 3])

=== Multi-Head Self-Attention ===
Output shape: torch.Size([1, 3, 4])
Attention weights shape: torch.Size([1, 2, 3, 3])

---

📌 Key Takeaways:

* **Self-attention allows each word to dynamically decide which other words matter.**
* **Multi-head attention enables multiple perspectives over the same input.**
* **This mechanism is the foundation of all modern LLMs.**
* **Understanding attention deeply is crucial for working with:**
* *1.Transformers.*
* *2.RAG systems.*
* *3.LLM-based agents.*

---

 Reference: 

Vaswani et al., Attention Is All You Need, NeurIPS 2017

---

 Final Note: 

This project focuses on clarity and understanding, not scale. It is intended as a learning-oriented, foundational implementation that bridges research papers and real-world Transformer-based systems.
