---
title: rag pipeline
source: sources/rag-pipeline.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# RAG Pipeline (Retrieval-Augmented Generation)

Retrieve relevant documents → augment prompt → generate. The most deployed pattern in production AI. Verified end-to-end with Ollama nomic-embed-text + MiniMax M2.7.

## The RAG Pattern

```
Query → Embed query → Vector search → Top-k chunks → Augment prompt → LLM Generate → Answer
```

## Verified Pipeline

### 1. Chunk and Index
```python
# Fixed-size chunking
chunks = chunk_text_fixed(document, chunk_size=50, overlap=10)
# Embed all chunks
chunk_embeddings = embed_texts(chunks)  # Ollama nomic-embed-text
# Build index
index = VectorIndex()
for chunk, emb in zip(chunks, chunk_embeddings):
    index.add(emb, chunk)
```

### 2. Retrieve
```python
query_emb = embed_texts([query])[0]
results = index.search(query_emb, top_k=3)
context_docs = [r["text"] for r in results]
```

### 3. Augment and Generate
```python
prompt = (
    "Use the provided context to answer the question.\n\n"
    + "\n\n".join(f"Document: {d}" for d in context_docs)
    + f"\n\nQuestion: {query}\nAnswer:"
)
response = minimax.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)
```

## RAG vs No-RAG: Key Comparison

| Scenario | No-RAG | RAG |
|---------|--------|-----|
| "What is Python 3.12's new feature?" | Generic answer, mentions multiple features, some hallucinated | Correctly mentions "type parameter syntax for generic classes" |

**Verified:** No-RAG gave a generic hallucinated response. RAG correctly retrieved and used the exact context.

## Why RAG Beats Fine-Tuning

- Cost: $0.01-$0.10 per query vs $1000+ per fine-tune run
- Freshness: re-index docs in minutes vs retrain model
- Auditability: show exact retrieved passages vs cannot trace source
- Hallucination: grounded in retrieved docs vs hallucinates freely

## Embedding Models

| Model | Dimensions | Provider | Cost |
|-------|-----------|----------|------|
| nomic-embed-text | 768 | Ollama (local) | Free |
| text-embedding-3-small | 1536 | OpenAI | $0.02/1M |
| BGE-M3 | 1024 | Open source | Free |

## Related Concepts

- [[Embeddings and Vector Search]] — embedding generation and similarity metrics
- [[Advanced RAG]] — hybrid search, BM25, reranking
- [[Context Engineering]] — managing retrieved context in the window
- [[Structured Outputs]] — extracting structured data from RAG responses
