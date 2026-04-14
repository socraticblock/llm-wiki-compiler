---
title: embeddings and vector search
source: sources/embeddings-and-vector-search.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# Embeddings and Vector Search

Text to dense vectors where geometric proximity encodes semantic similarity. Verified with Ollama nomic-embed-text (768d, unit-normalized).

## Verified Setup

**Ollama nomic-embed-text:**
```python
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = ollama.embeddings.create(model="nomic-embed-text", input=["text"])
embedding = response.data[0].embedding  # 768 dimensions, unit-normalized
```

- Dimension: 768
- Unit-normalized: Yes (cosine = dot product)
- Dimension: 768
- Generation time: ~20ms per text on CPU

## Similarity Metrics

### Cosine Similarity (default)
```python
def cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))  # both unit-normalized
```
Range: -1 to 1. Best for texts of different lengths.

### Dot Product
Identical to cosine for normalized vectors. Faster (no norm computation).

### Euclidean Distance
Straight-line distance. Lower = more similar. Use for clustering.

## Semantic Search Pipeline

```python
# 1. Index documents
doc_embeddings = embed_texts(documents)

# 2. Search
query_emb = embed_texts([query])[0]
scores = [cosine_similarity(query_emb, e) for e in doc_embeddings]
top_k = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)[:5]
```

**Verified results:**
- "I can't remember my password" → #1: "How do I reset my password?" (0.80) ✅
- "Python web framework for APIs" → #1: "FastAPI..." (0.82) ✅
- "Database with vector search" → #1: "pgvector..." (0.70) ✅

## Chunking Strategies

### Fixed-Size (simple, lossy)
```python
def chunk_text_fixed(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start+chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
```

### Sentence-Based (better coherence)
```python
def chunk_sentences(text, max_words=50):
    sentences = [s.strip() for s in text.replace(".", ".\n").split("\n")]
    chunks, current = [], []
    word_count = 0
    for sent in sentences:
        sw = len(sent.split())
        if word_count + sw > max_words and current:
            chunks.append(" ".join(current))
            current = []
            word_count = 0
        current.append(sent)
        word_count += sw
    if current:
        chunks.append(" ".join(current))
    return chunks
```

**Verified: Sentence-based produces more coherent chunks than fixed-size.** Fixed-size cuts mid-sentence.

## Semantic Search Quality

| Query | Top Result | Score |
|-------|-----------|-------|
| "I can't remember my password" | "How do I reset my password?" | 0.80 |
| "I forgot my login credentials" | "How do I reset my password?" | 0.77 |
| "Python web framework" | "FastAPI is a Python web framework..." | 0.86 |
| "Database vector search" | "pgvector is PostgreSQL for vectors" | 0.70 |

## OpenAI-Compatible Embedding API

MiniMax does not provide OpenAI-compatible embeddings. Use Ollama for local embeddings:

```python
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

Alternative: use OpenAI's `text-embedding-3-small` (1536d, $0.02/1M tokens).

## Related Concepts

- [[RAG Pipeline]] — combining embeddings with LLM generation
- [[Advanced RAG]] — hybrid search, BM25, reciprocal rank fusion
- [[Context Engineering]] — retrieved context management
