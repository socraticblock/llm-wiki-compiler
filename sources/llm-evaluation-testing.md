---
title: llm evaluation and testing
source: sources/llm-evaluation-testing.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# LLM Evaluation & Testing

Automated evaluation frameworks for LLM applications. Verified with MiniMax M2.7 as judge.

## LLM-as-Judge

Use an LLM to evaluate another LLM's responses against a rubric.

```python
def llm_judge(question, answer, rubric="Is the answer accurate, complete, and helpful?"):
    prompt = (
        f"Score this answer 1-5.\nQuestion: {question}\n"
        f"Answer: {answer}\nRubric: {rubric}\n"
        "Respond with JSON: {score: int, reasoning: str}"
    )
    response = minimax.chat.completions.create(
        model="MiniMax-M2.7",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(strip_think_block(response.choices[0].message.content))
```

**Verified results:**
- Good answer to "Python 3.12 feature?" scored high (informative, specific)
- Bad answer ("many features including...") scored 2/5 (vague, incomplete)
- Good answer to "Capital of France?" scored 5/5
- Bad answer ("France is a country in Europe") scored 1/5 (incorrect focus)

Key: Always apply `strip_think_block()` before JSON parsing on MiniMax.

## Automated Metrics

### Exact Match
```python
def exact_match(pred, expected):
    return pred.strip().lower() == expected.strip().lower()
```
**Limitation:** Too strict. "42" vs "The answer is 42" fails. Use keyword presence instead.

### Keyword/Rouge-like
```python
def contains_keywords(response, keywords):
    return any(kw.lower() in response.lower() for kw in keywords)
```
**Better for production:** Accepts variation in phrasing.

## Regression Testing

Run a test suite on every code change:

```python
test_suite = [
    {"id": "t1", "query": "What is 2+2?", "expected_keywords": ["4"]},
    {"id": "t2", "query": "Capital of Japan?", "expected_keywords": ["Tokyo"]},
]

for tc in test_suite:
    response = call_llm(tc["query"])
    passed = contains_keywords(response, tc["expected_keywords"])
    log_result(tc["id"], passed)
```

**Verified: 4/4 regression tests passed** using keyword matching.

## Evaluation Framework Design

1. **Define ground truth**: input-output pairs + rubrics
2. **Automated scoring**: LLM-as-judge for quality, keyword match for factual
3. **Regression suite**: run on every change, block deployment on degradation
4. **Confidence intervals**: LLM outputs are stochastic — run N samples, report mean ± std
5. **Human spot-check**: LLM-as-judge is not perfect — human review on edge cases

## MiniMax Evaluation Pipeline

```python
# 1. Build eval dataset
eval_set = [
    {"input": "...", "expected": "...", "rubric": "..."},
]

# 2. Run all evaluations
for item in eval_set:
    response = call_llm(item["input"])
    if item.get("expected"):
        exact = exact_match(response, item["expected"])
    if item.get("rubric"):
        judgment = llm_judge(item["input"], response, item["rubric"])

# 3. Report aggregate metrics
print(f"Accuracy: {correct}/{total}")
print(f"Avg LLM-as-Judge score: {sum(scores)/len(scores):.2f}/5")
```

## Related Concepts

- [[AI Engineering LLM Engineering Patterns]] — prompting fundamentals
- [[Structured Outputs]] — parsing LLM responses
- [[Guardrails]] — safety testing
