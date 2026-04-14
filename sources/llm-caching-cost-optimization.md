---
title: llm caching and cost optimization
source: sources/llm-caching-cost-optimization.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# LLM Caching & Cost Optimization

Reducing API costs through caching strategies and model routing. Verified with MiniMax M2.7 + Ollama.

## Cost Anatomy

A single LLM call has 5 cost components:
- System prompt tokens (billed every request)
- Retrieved context tokens
- User message tokens
- Output tokens
- Embedding tokens (for RAG)

## Exact-Match Cache

```python
import hashlib

class ExactCache:
    def __init__(self):
        self.store = {}

    def make_key(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text):
        return self.store.get(self.make_key(text))

    def set(self, text, result):
        self.store[self.make_key(text)] = result
```

**Verified:**
- Cache miss: ~12,300ms (API call)
- Cache hit: ~0.002ms
- Speedup: 6,000,000x

## Semantic Cache

For near-duplicate queries, use embedding similarity:

```python
class SemanticCache:
    def __init__(self, similarity_threshold=0.90):
        self.entries = []  # (embedding, text, result)
        self.threshold = similarity_threshold

    def get(self, text, embedding):
        for emb, orig, result in self.entries:
            sim = cosine_sim(embedding, emb)
            if sim >= self.threshold:
                return result, sim
        return None, 0.0
```

**Verified:**
- "How do I reset my password?" → cache hit on "I can't reset my password" (0.91 similarity)
- "What is the weather today?" → cache miss (0.0 similarity to password queries)

Threshold tuning: 0.90 is aggressive but catches paraphrases. Higher = fewer hits but higher confidence.

## Cost Calculation

```python
COSTS = {
    "MiniMax-M2.7": {"input": 0.002, "output": 0.002},  # per 1K tokens
    "qwen3:8b": {"input": 0.0, "output": 0.0},  # local, free
}

def calc_cost(model, prompt_tokens, completion_tokens):
    rates = COSTS[model]
    return prompt_tokens * rates["input"] / 1000 + completion_tokens * rates["output"] / 1000
```

**Verified batch:** 3 calls × ~500-1200 prompt + ~150-300 completion = $0.0063 total. Extrapolated 1000 calls: $6.30.

## Model Routing

Route to cheap vs expensive model based on query complexity:

```python
SIMPLE_KEYWORDS = ["what is", "who is", "define", "capital of"]
COMPLEX_KEYWORDS = ["explain", "analyze", "compare", "write code", "debug"]

def route_query(query):
    q = query.lower()
    if any(kw in q for kw in SIMPLE_KEYWORDS):
        return "cheap"  # e.g., MiniMax
    elif any(kw in q for kw in COMPLEX_KEYWORDS):
        return "expensive"  # e.g., GPT-4o
    return "medium"
```

**Verified routing:**
- "What is Python?" → cheap
- "Compare FastAPI vs Django" → expensive
- "Capital of France?" → cheap

## Tiered Caching Strategy

| Query Type | Cache Strategy | Threshold |
|-----------|---------------|-----------|
| Identical repeated | Exact match | 100% |
| Paraphrase | Semantic | 0.90+ |
| System prompt | Prefix caching | (model handles) |
| RAG context | Fixed context cache | (vector DB handles) |

## Key Insight

40-60% of production LLM queries are near-duplicates. A semantic cache with 0.90 similarity threshold catches most paraphrases at near-zero cost.

## Related Concepts

- [[RAG Pipeline]] — caching retrieved documents
- [[Context Engineering]] — token budgeting
- [[LLM Evaluation and Testing]] — cost tracking in eval
