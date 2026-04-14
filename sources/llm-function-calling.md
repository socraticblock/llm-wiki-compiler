---
title: llm function calling
source: sources/llm-function-calling.md
ingestedAt: "2026-04-14T09:00:00Z"
---

# LLM Function Calling & Tool Use

LLMs generate structured JSON describing which function to call and with what arguments. Verified with MiniMax M2.7.

## The Tool Loop

```
User → Model (with tool schemas) → Tool Call JSON → Execute function → Return result → Model (with result) → Final answer
```

## Tool Schema Definition

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name e.g. 'Tokyo'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "default": "celsius"
        }
      },
      "required": ["city"]
    }
  }
}
```

## Verified Implementation (MiniMax M2.7)

```python
response = minimax.chat.completions.create(
    model="MiniMax-M2.7",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[TOOL_SCHEMA],
    tool_choice="auto"
)

msg = response.choices[0].message
if msg.tool_calls:
    tc = msg.tool_calls[0]
    args = json.loads(tc.function.arguments)
    result = execute_tool(tc.function.name, args)
    # Feed back to model...
```

## Verified Capabilities

| Capability | Status | Notes |
|-----------|--------|-------|
| Single tool call | ✅ | Works reliably |
| Parallel tool calls | ✅ | 2 simultaneous calls (weather + time) |
| Enum constraints | ✅ | Model respects enum parameter options |
| No-tool fallback | ✅ | Model answers directly when no tool needed |
| JSON arguments | ✅ | Clean JSON in function.arguments |

## Multi-Turn Agent Loop

```python
def tool_call_loop(user_query, max_turns=5):
    messages = [{"role": "user", "content": user_query}]
    for turn in range(max_turns):
        response = call_model(messages)
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content  # final answer
        for tc in msg.tool_calls:
            result = execute_tool(tc.function.name, tc.function.arguments)
            messages.append({"role": "assistant", "tool_calls": [tc]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })
```

## Tool Schema Design Principles

1. **Clear descriptions**: Help the model decide when to call the tool
2. **Typed parameters**: `enum` for constrained values, `string`/`number`/`boolean` for others
3. **Required fields**: Only mark truly required fields as required
4. **Default values**: Provide defaults to reduce mandatory parameters
5. **Descriptions on fields**: Tell the model what values are valid

## Edge Cases

- **Model hallucinates tool arguments**: Use function calling (schema-constrained) over JSON mode for structured tool invocation
- **Tool loop infinite recursion**: Set `max_turns` limit
- **Tool execution failure**: Return error as string, let model decide how to handle

## Related Concepts

- [[Structured Outputs]] — JSON mode, schema validation
- [[AI Engineering LLM Engineering Patterns]] — CoT, few-shot prompting
- [[Guardrails]] — tool call safety, output validation
