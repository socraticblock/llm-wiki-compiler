---
title: advanced rag techniques
source: sources/advanced-rag-techniques.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# Advanced RAG Techniques

Beyond basic vector search: hybrid search, BM25 keyword matching, reciprocal rank fusion, and query transformation. Verified with Ollama nomic-embed-text + MiniMax M2.7.

## BM25 Keyword Search

Bi-encoder semantic search misses exact matches. BM25 excels at exact term matching.

```python
class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, documents):
        self.corpus = [doc.lower().split() for doc in documents]
        # compute IDF for each term

    def score(self, query, doc_idx):
        # BM25 formula: IDF * (tf * (k1+1)) / (tf + k1*(1-b+b*dl/avgdl))
        ...

    def search(self, query, top_k=5):
        return sorted([(i, self.score(query, i)) for i in range(len(self.corpus))],
                      key=lambda x: x[1], reverse=True)[:top_k]
```

**Verified BM25 results:**
- Query "Python web framework" → FastAPI score: 4.38 (highest)
- Query "TypeScript JavaScript" → TypeScript score: 3.22, React: 1.21
- Query "Docker Kubernetes" → Kubernetes: 1.99, Docker: 1.88

## Reciprocal Rank Fusion (RRF)

Combine semantic (vector) and keyword (BM25) rankings:

```python
def rrf_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for rank, (idx, vec_score) in enumerate(vector_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, bm25_score) in enumerate(bm25_results):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

k=60 is standard (prevents top result from dominating).

**Verified RRF fusion:**
- Query "Python web framework": FastAPI ranked #1 (top in both vector AND BM25)
- Fusion balances semantic similarity + exact keyword match

## Query Transformation

### Multi-Query Expansion

```python
prompt = (
    "Generate 3 different versions of this search query using different words.\n"
    "Return JSON with field 'queries' (list of strings).\n\n"
    f"Query: {original_query}"
)
# Use MiniMax JSON mode to get expanded queries
```

**Verified:** "Python async programming" → ["asynchronous programming using Python", "Python async/await tutorial", "how to use async in Python"]

### HyDE (Hypothetical Document Embeddings)
Generate a hypothetical document answering the query, then embed that to retrieve real documents.

### Step-Back Prompting
Ask a more general version of the question to retrieve broader context.

## Reranking (Cross-Encoder)

Bi-encoder retrieves fast (top-50), cross-encoder reranks precisely (top-5).

```python
# Retrieve top-50 with bi-encoder
candidates = vector_search(query, top_k=50)
# Rerank top-50 with cross-encoder
reranked = cross_encoder.rerank(query, candidates, top_k=5)
```

Open-source rerankers: BGE-reranker-v2, Jina Reranker v2.

## Chunking Strategies Comparison

| Strategy | Quality | Best For |
|----------|---------|----------|
| Fixed-size | Decent | Unstructured text |
| Sentence-based | Good | Articles, structured docs |
| Semantic | Best | High-value retrieval |

**Verified:** Sentence-based chunking preserves complete thoughts. Fixed-size cuts mid-sentence.

## Related Concepts

- [[RAG Pipeline]] — basic retrieval and generation
- [[Embeddings and Vector Search]] — vector similarity fundamentals
- [[Context Engineering]] — managing retrieved context
