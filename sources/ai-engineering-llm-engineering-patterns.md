---
title: ai engineering llm engineering patterns
source: sources/ai-engineering-llm-engineering-patterns.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# AI Engineering: LLM Engineering Patterns

A compiled reference from the AI Engineering from Scratch curriculum (Phase 11), verified with MiniMax M2.7 and Ollama nomic-embed-text.

## Prompt Engineering Patterns

### Persona Pattern
Activate specific expert distributions in the model's training data.

```
You are {role} with {experience}.
Your communication style is {style}.
You prioritize {priority}.
Task: {task}
```

Verified: Specific role framing produces measurably better responses on MiniMax M2.7.

### Few-Shot Pattern
Provide concrete examples to anchor output format and style.

```
Examples:
Input: "food was amazing but service slow"
Output: {"sentiment": "mixed", "food": "positive", "service": "negative"}

Now process:
Input: "great pasta, a bit pricey"
Output:
```

Verified: Temp=0.0 for format-sensitive tasks. Works reliably on MiniMax M2.7.

### Chain-of-Thought (CoT)
Force explicit reasoning steps before the final answer.

```
Think step by step.
Problem: {problem}
Show your reasoning first, then state the answer.
```

Verified: "Let's think step by step" consistently improves accuracy on math problems. MiniMax M2.7 correctly follows reasoning chains.

### Self-Consistency
Sample N reasoning paths at temperature>0, take majority vote.

- N=3 minimum for meaningful vote
- N=5 captures most benefit
- Temperature 0.7 — not 0 (would be identical), not 1.0 (too random)

Verified: 5/5 samples agreed on a straightforward math problem (100% confidence). Works on MiniMax.

### Template Fill Pattern
Constrain output to a specific structure with named fields.

```
Extract information and fill the template:
Name: [name]
Company: [company]
...
Fill every field. If not available, write 'N/A'.
```

Verified: Clean extraction at temp=0.0 on MiniMax.

### Guardrail Pattern
Explicit scope boundaries with refusal message.

```
You are a {role}.
ONLY answer questions about {domain}.
If outside scope, say: 'This is outside my scope.'
```

Verified: Model stayed scoped, guided vs. gave full solutions. Works on MiniMax.

### Boundary Pattern
Hard boundary with exact refusal message.

```
If request is within scope, help fully.
If outside scope, respond EXACTLY with:
'{refusal_message}'
Do not attempt out-of-scope questions.
```

Verified: Out-of-scope → exact refusal, no drift. Works on MiniMax.

## MiniMax-Specific Gotchas

### Think Block Prepend
MiniMax M2.7 outputs a reasoning block before JSON responses.

```python
def strip_think_block(raw_text):
    json_start = raw_text.rfind('{')
    return raw_text[json_start:] if json_start != -1 else raw_text.strip()
```

Always apply this before JSON parsing.

### JSON Mode
Use `response_format={"type": "json_object"}` for guaranteed valid JSON.
Flat schemas work. Nested schemas may fail to parse on MiniMax.

### Function Calling
Tool_calls work reliably on MiniMax M2.7. Use enum constraints for parameter values.

## Related Concepts

- [[Structured Outputs]] — JSON mode, schema validation, constrained decoding
- [[Embeddings and Vector Search]] — semantic similarity, cosine similarity
- [[Context Engineering]] — token budgets, lost-in-the-middle, context assembly
- [[RAG Pipeline]] — retrieval-augmented generation combining embeddings + LLM
- [[Function Calling]] — tool use, multi-turn agents
