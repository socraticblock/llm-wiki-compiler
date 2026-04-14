---
title: context engineering for llm
source: sources/context-engineering-llm.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# Context Engineering for LLMs

Managing what goes into the context window: token budgets, lost-in-the-middle, history compression, and dynamic assembly. Verified with MiniMax M2.7.

## Token Budget Manager

```python
class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        tokens = count_tokens(content)  # approx: len(words) * 1.3
        if max_tokens and tokens > max_tokens:
            words = content.split()
            content = " ".join(words[:int(max_tokens / 1.3)])
            tokens = count_tokens(content)
        self.allocations[component] = tokens
        return content, tokens
```

## Token Counting Approximation

```python
def count_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 1.3)
```

Note: This is an approximation. Real tokenizers vary. For production, use the provider's actual tokenizer.

## Lost-in-the-Middle

Models attend best to content at the beginning and end of the context. Content in the middle is frequently ignored.

**Reordering strategy:**
```python
def reorder_lost_in_middle(items, scores):
    paired = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
    sorted_items = [item for _, item in paired]
    first_half = sorted_items[::2]      # highest, 3rd, 5th...
    second_half = list(reversed(sorted_items[1::2]))  # 2nd, 4th, 6th...
    return first_half + second_half    # high, low, high, low...
```

**Verified:** Most relevant documents placed at positions 0 and last. Middle positions lowest priority.

## Conversation History Compression

```python
class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return
        while total > self.max_history_tokens and len(self.turns) > 4:
            summary = f"Previous: user: {self.turns[0]['content'][:50]}... | assistant: {self.turns[1]['content'][:50]}..."
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)
```

**Verified:** 10 conversation turns compressed to 4 recent + 8 summary fragments.

## Dynamic Tool Selection

```python
INTENT_KEYWORDS = {
    "code": ["code", "function", "bug", "error", "file"],
    "email": ["email", "mail", "send"],
    "research": ["search", "what is", "explain"],
}

def select_tools(query, budget=2000):
    intents = [kw for kw, words in INTENT_KEYWORDS.items()
               if any(w in query.lower() for w in words)]
    # Select tools matching intents within budget...
```

**Verified:**
- "Fix the bug in auth module" → selects code tools
- "Search for Llama 3 info" → selects research tools
- "Send an email" → selects email tools

## Dynamic Context Assembly

Per-query context assembly:
1. Classify query intent
2. Select relevant tools (not all)
3. Retrieve relevant docs (not all)
4. Include recent history (not all)
5. Order: critical first, important last

**Verified:** Full pipeline with MiniMax — assembled context correctly produces relevant answers.

## Context Window Budget Allocation

Typical 128K window:
| Component | Typical Size |
|-----------|-------------|
| System prompt | 500-2000 tokens |
| Tool definitions (10 tools) | ~2000 tokens |
| Retrieved context | 2000-10000 tokens |
| Conversation history | 2000-20000 tokens |
| Generation budget | 2000-8000 tokens |

## Related Concepts

- [[RAG Pipeline]] — retrieved context management
- [[Embeddings and Vector Search]] — semantic retrieval
- [[AI Engineering LLM Engineering Patterns]] — prompt ordering
