---
title: structured outputs with llm
source: sources/structured-outputs-with-llm.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# Structured Outputs with LLMs

Techniques for getting typed, validated data from language models. Verified with MiniMax M2.7 and OpenAI-compatible APIs.

## The Structured Output Spectrum

| Level | Method | Reliability | Guarantee |
|-------|--------|-------------|-----------|
| Prompt-based | "Return JSON" | ~90% | None |
| JSON Mode | `response_format={type:"json_object"}` | ~99% | Valid JSON only |
| Function Calling | Tool use with schema | ~99% | Schema-compliant |
| Constrained Decoding | Token masking | 100% | Grammar-compliant |

## JSON Mode (MiniMax)

```python
response = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[{"role": "user", "content": "Extract product info..."}],
    response_format={"type": "json_object"},
    max_tokens=1500
)
parsed = json.loads(strip_think_block(response.choices[0].message.content))
```

**Verified findings:**
- Simple flat schemas work reliably
- Nested schemas can fail to parse on MiniMax — use flat structures
- Always apply `strip_think_block()` before JSON parsing
- Temperature=0.0 for deterministic extraction

## Function Calling (MiniMax)

```python
response = client.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[{"role": "user", "content": "Extract: Apple MacBook costs $2499"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "extract_product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string"},
                    "price": {"type": "number"},
                    "in_stock": {"type": "boolean"}
                },
                "required": ["product", "price", "in_stock"]
            }
        }
    }],
    tool_choice="auto"
)
tc = response.choices[0].message.tool_calls[0]
args = json.loads(tc.function.arguments)
```

**Verified: Function calling works reliably on MiniMax M2.7.** Parallel tool calls also work (tested with weather + time queries simultaneously).

## Schema Validation

A simple Pydantic-style validator catches type and constraint errors:

```python
def validate(data, schema):
    errors = []
    if "price" in schema.get("required", []) and "price" not in data:
        errors.append("Missing required field: price")
    if "price" in data and not isinstance(data["price"], (int, float)):
        errors.append("Price must be a number")
    if "price" in data and isinstance(data["price"], (int, float)) and data["price"] < 0:
        errors.append("Price must be non-negative")
    return errors
```

## Extraction Pipeline Pattern

```
User Input → Prompt (with schema) → MiniMax JSON mode → strip_think_block → validate → retry if needed → parsed object
```

Key principles:
- Trust JSON mode over manual parsing
- Validate schema compliance after parsing
- Retry with error feedback on validation failure
- For enum fields: use tool calling (enum constraint enforced) rather than JSON mode (prompt-based)

## Common Failure Modes

1. **Nested schemas**: MiniMax JSON mode struggles with nested objects — flatten schemas
2. **Hallucinated values**: schema validation can't catch invented data — use function calling for high-stakes extraction
3. **Enum violations**: JSON mode prompt-based enums are not enforced — use function calling's enum support
4. **Think block prepend**: always strip before parsing

## Related Concepts

- [[AI Engineering LLM Engineering Patterns]] — CoT, few-shot, persona patterns
- [[Function Calling]] — tool use, multi-turn agent loops
- [[Guardrails]] — input/output validation layers
