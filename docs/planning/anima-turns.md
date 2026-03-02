## The Problem

Getting local models to use tools reliably is the central challenge of running an agentic loop on Ollama. Cloud APIs (OpenAI, Anthropic) have been fine-tuned specifically for tool calling — they emit structured tool invocations cleanly, chain calls naturally, and stop when done. Local models do none of this consistently. They hallucinate tool names, emit malformed JSON, forget to call tools when they should, call tools when they shouldn't, and loop endlessly without producing a final answer.

This document explores what happens on each iteration of Anima's tool loop, what works, what breaks, and what we can do about it. The focus is narrow: the agentic loop itself — how the LLM sees tools, how it emits calls, how results come back, and how the loop terminates. System prompt design, memory injection, and persona tuning are out of scope.

## Anatomy of a Turn

A single iteration of the tool loop looks like this from the LLM's perspective:

```
Messages the LLM receives:
┌─────────────────────────────────────────────┐
│ system: "You are..."                        │
│ user: "Read main.rs and fix the bug"        │
│ assistant: [tool_call: read_file(main.rs)]  │
│ tool: "[Tool Result for read_file]\n..."    │
│ assistant: [tool_call: edit_file(...)]      │  ← history from prior iterations
│ tool: "[Tool Result for edit_file]\n..."    │
│ user: "[Tool call 4 of 10 — consider...]"  │  ← budget nudge (if applicable)
└─────────────────────────────────────────────┘

LLM generates:
┌─────────────────────────────────────────────┐
│ Either: tool_calls: [...]                   │  ← loop continues
│ Or:     "Done. I fixed the null check."     │  ← loop terminates
└─────────────────────────────────────────────┘
```

The harness does exactly one thing per iteration: call the LLM, check whether the response contains tool calls, and either execute them (loop continues) or return the text (loop ends). Everything else — context management, deduplication, budget nudges — is bookkeeping around this core decision.

Anima has two implementations of this loop. The agent's internal loop (`think_with_options_inner` in `agent.rs`) holds history in memory and runs self-contained. The daemon's loop (`run_tool_loop` in `daemon.rs`) persists each turn to SQLite, clears the agent's in-memory history between iterations, and reconstructs context from the database. In daemon mode — which is the production path — the database is the source of truth, not the agent's internal state.

## Tool Presentation

### Native Mode (`tools = true`)

When the model supports native tool calling, Anima converts all allowed tools into `ToolSpec` objects and passes them through the API's tools field. The LLM provider handles the rest — OpenAI gets `tool_choice: "auto"`, Anthropic gets `input_schema`, Ollama gets the OpenAI-compatible format.

The conversion pipeline: `tools.toml` defines tools with flat param maps (`{"path": "string", "content": "string"}`). These are expanded into JSON Schema via `convert_params_to_json_schema()` in `tool_registry.rs`, producing proper `{"type": "object", "properties": {...}, "required": [...]}` schemas. All allowed tools are injected every turn — there's no per-query filtering in native mode.

This works well with cloud models. With Ollama it's inconsistent. Some models (Qwen 2.5 Coder, Llama 3.x with tool support) handle native tool calling adequately. Others ignore the tools field entirely or emit tool calls with wrong parameter names.

### Tool-Block Mode (`tools = false`)

When native tool calling is unreliable, Anima falls back to tool-block mode. Instead of passing tools through the API, the harness selects 3-5 relevant tools by keyword-matching the user's query and injects them as markdown into the system context:

```
**Available tools:**
- `list_tools` — List all available tools. Params:
- `read_file` — Read a file. Params: `path` (string)
- `edit_file` — Edit a file. Params: `path` (string), `old` (string), `new` (string)
- `shell` — Run a shell command. Params: `command` (string)

**Tool call format:**
To use a tool, wrap a JSON object in <tool> tags:

<tool>
{"tool": "TOOL_NAME", "params": {"PARAM": "VALUE"}}
</tool>

To respond without tools, write your answer normally (no <tool> tags).
Only use tools listed above. Do not invent tool names.
```

The LLM calls tools by emitting a `<tool>` XML tag containing JSON:

```
<tool>
{"tool": "read_file", "params": {"path": "src/main.rs"}}
</tool>
```

The harness extracts this with a regex (`(?s)<tool>\s*(\{.*?\})\s*</tool>`), executes the tool, and injects the result as the next message. Using `<tool>` tags instead of fenced code blocks avoids collisions with markdown code blocks in the model's natural output and allows backticks inside parameter values. If extraction fails but a `<tool>` tag is detected, a format error is fed back to the LLM so it can retry. The keyword filtering is the key difference from native mode — the LLM only sees tools relevant to the current query, reducing the chance of hallucinating irrelevant calls.

### Observations

**Fewer tools = better accuracy.** Local models degrade rapidly when presented with more than 5-6 tools. They start confusing parameter schemas, calling the wrong tool for the job, or inventing tools that don't exist. tool-block mode's keyword filtering (3-5 tools per query) works in its favor here.

**Tool descriptions matter more than schemas.** Cloud models read JSON Schema carefully. Local models mostly ignore it and rely on the description text to figure out what parameters to pass. A description like "Read a file from disk" with params `{"path": "string"}` works. A terse name like `read_file` with a complex schema doesn't.

**Native mode gives better chaining.** When native tool calling works at all, it produces better multi-step behavior than tool-block mode. The model seems to understand the tool-call/tool-result message structure more naturally than the "emit JSON, read result, emit more JSON" pattern.

**The `list_tools` meta-tool is a good safety net.** In tool-block mode, the injected tool list always includes `list_tools`, which returns all available tools. This lets the LLM discover tools that weren't selected by keyword matching. It adds one extra round-trip but prevents the model from being stuck without the right tool.

## Tool Call Extraction

### Native Path

In native mode, the LLM provider's API returns structured tool calls directly. Anima parses them per-provider:

- **OpenAI/Ollama**: Tool calls arrive as `message.tool_calls[]` with `function.name` and `function.arguments` (a JSON string). The arguments string is parsed into `serde_json::Value`. If parsing fails, the `ToolCall` is created with `arguments: Value::Null` and `parse_error` set to the error message — the call isn't dropped, it's fed back as an error.
- **Anthropic**: Tool calls arrive as `content[]` blocks with `type: "tool_use"`. Arguments are already a JSON object in `block["input"]`, no string parsing needed.

### XML Fallback

Some Ollama models don't use the native tool calling protocol but instead emit XML tool invocations in their response text:

```xml
<function=read_file><parameter=path>src/main.rs</parameter></function>
```

Anima handles this with `apply_xml_tool_fallback()` in `llm.rs`: if the native `tool_calls` array is empty but the response text contains these XML patterns, they're extracted via regex and converted to `ToolCall` objects. The XML is stripped from the content. This is a pragmatic fallback for models that were trained on this format (some Llama variants).

### Tool-Block Path

In tool-block mode, extraction is a single regex:

```rust
r"(?s)<tool>\s*(\{.*?\})\s*</tool>"
```

This captures a JSON object inside `<tool>...</tool>` XML tags. The `(?s)` flag enables dotall so `.*?` matches newlines inside the JSON body. The JSON must contain a `"tool"` key (string) and optionally a `"params"` key (object). If the regex doesn't match, or the JSON is malformed, or the `"tool"` key is missing, extraction returns `None`. If a `<tool>` tag was present but extraction failed, a format error is fed back to the LLM instead of silently terminating the loop.

### Observations

**Parse errors now feed back.** In both modes, Anima feeds parse errors back to the LLM. In tool-block mode, if a `<tool>` tag is detected but extraction fails (malformed JSON, missing `"tool"` key, unclosed tag), the harness sends a format error with the expected format, giving the model a chance to retry.

**`<tool>` tags reduce false positives.** The old fenced code block format (`\`\`\`json`) collided with markdown code blocks in the model's natural output. Using `<tool>` tags is unambiguous — models don't emit `<tool>` in commentary. The `"tool"` key check provides a second layer of validation.

**Multiple tool calls per turn.** Native mode supports multiple tool calls in a single response — OpenAI and Anthropic both allow this. tool-block mode only extracts the first match. For local models this is usually fine (they rarely emit multiple calls successfully), but it's a structural limitation.

## Result Feedback

How the LLM sees tool results after execution:

### Native Mode

Tool results are stored as messages with `role: "tool"` and `tool_call_id` matching the original call's ID. The content is formatted as:

```
[Tool Result for read_file]
1: fn main() {
2:     println!("hello");
3: }
```

Or on error:

```
[Tool Error for read_file]
File not found: /nonexistent.rs
```

The `[Tool Result for X]` / `[Tool Error for X]` prefix is important — it tells the model which tool produced the output, especially when multiple tools were called in one turn.

### Tool-Block Mode

Results are injected differently. The tool result is stored as a separate message in the conversation and becomes part of the history the LLM sees on the next iteration. There's no `tool_call_id` linkage — the model has to infer which tool produced which output from the content itself.

### Truncation

Tool results are truncated to 10% of the model's context window (by character count, estimated at 4 chars per token). When truncated, the **tail** is kept (not the head), and a header is prepended:

```
[output truncated: 1.2 MB, showing last 25.6 KB]

...remaining output here...
```

Keeping the tail is the right default for most tools: shell output has the exit code and summary at the end, file reads have the most recently relevant section at the bottom, and error messages appear at the end of compiler output.

### Observations

**Result format should be minimal.** Local models get confused by verbose result formatting. The `[Tool Result for X]` prefix is enough context. Adding XML wrappers, extra metadata, or structured formatting around results degrades performance — the model spends tokens parsing the wrapper instead of reasoning about the content.

**Truncation thresholds need tuning per model.** The 10% rule works for 32K-128K context windows. For smaller contexts (8K), 10% is only ~3.2K characters — too small for many file reads or shell outputs. For very large contexts, 10% might be wastefully generous. A floor (e.g., 4K minimum) and ceiling would be better.

**Error results are more useful than error non-results.** When a tool fails, the LLM benefits from seeing the error message, the tool name, and ideally the expected parameter schema. Anima does this well in native mode (the parse error includes `Expected parameters: ...`). The pattern should be consistent across both modes.

## Loop Termination

The loop ends when the LLM produces a response with no tool calls. In native mode, this means `response.tool_calls` is empty. In tool-block mode, this means the response contains no `<tool>` tags (or only a malformed one that triggers error feedback first).

### The Signals

The LLM has no explicit "I'm done" signal. Termination is inferred from the absence of tool calls. This works well when the model is confident in its answer — it simply responds in natural language without emitting any tool invocations.

Additional termination conditions in the harness:

| Condition | Behavior |
|---|---|
| Max iterations reached | Returns `[Max iterations reached: N]` |
| Wall-clock deadline exceeded | Returns `[Response terminated: ...]` |
| Cancellation flag set | Returns `[Paused]` |
| Shutdown signal | Returns empty string |
| State machine enters `wait: true` | Returns response at that point |

### Budget Nudges

When the loop has consumed a significant fraction of its iteration budget, the harness appends a nudge to the conversation:

- At 50-79% of budget: `[Tool call 5 of 10 — consider responding if you have sufficient information]`
- At 80%+ of budget: `[Tool call 8 of 10 — approaching limit, provide your response now]`

These are injected as user messages, which means the model sees them as directives. This works — local models generally respond to explicit instructions about when to wrap up.

### Observations

**Local models loop more than cloud models.** A common pattern: the model reads a file, reads it again, reads a related file, re-reads the first file, and never produces a final answer. The budget nudge at 50% helps break this cycle, but it's a blunt instrument. The model has already wasted half its budget by the time the nudge appears.

**Format error feedback helps.** In tool-block mode, if the model emits a `<tool>` tag but the content doesn't parse, the harness now feeds back a format error instead of silently terminating. This addresses the biggest reliability gap — the model learns its tool call was malformed and can retry.

**Earlier nudges would help.** Instead of waiting until 50% to nudge, the harness could inject a lightweight signal after every tool result: `[Tools used: 3 of 10 remaining]`. This keeps the model aware of its budget without being directive, and might reduce unnecessary calls.

## Chaining

Multi-step tool use — read a file, edit it, run tests, fix failures — is where local models struggle most. The model needs to maintain a plan across turns, remember what it's already done, and decide what to do next based on accumulating results.

### What Works

**Short chains (2-3 steps).** "Read this file and fix the typo" works: read → edit → done. The model can hold a simple plan in context.

**Explicit sequencing in the prompt.** When the system prompt or user message describes the steps ("First read the file, then edit line 42, then run `cargo test`"), local models follow the sequence more reliably than when left to figure out the plan themselves.

**Tool results that suggest next steps.** When a shell command outputs "3 tests failed", the model naturally moves to read the failing test file. Results that are self-evidently incomplete drive the loop forward.

### What Breaks

**Long chains (5+ steps).** The model loses track of its plan. It re-reads files it already has in context, calls tools it's already called, or produces a final answer prematurely because it forgot there were more steps.

**Branching decisions.** "If the tests pass, deploy. If they fail, fix them." Local models struggle with conditional logic across tool calls. They tend to pick one branch and ignore the condition.

**Context pressure.** Each tool call and result adds to the context. By step 5-6 of a chain, a significant fraction of the context window is tool results. The model's attention degrades — it responds to the most recent result and forgets earlier context. Anima's deduplication helps (removing superseded read/write results), but it kicks in at 90% context fill, which is late.

### Context Management

Anima has a two-tier context management strategy:

1. **Deduplication** (`dedup_tool_results`): At 90% context fill, removes stale tool results — keeps only the last `read_file` per path, the last `write_file` per path, collapses identical shell commands. This is surgical: it removes information the model no longer needs without dropping anything it hasn't seen a more recent version of.

2. **Hard trim**: If dedup doesn't bring usage below 90%, the harness keeps the system message and the most recent messages that fit in 30% of the context window. Everything else is dropped. This is destructive but prevents the model from hitting the context limit and producing garbage.

### Observations

**Dedup should run earlier.** At 90%, the model has already been reasoning with a polluted context for several turns. Running dedup at 60-70% — or even proactively after every tool result — would keep the context clean throughout the chain. The cost is minimal (it's a linear scan over messages).

**Plan injection would help chaining.** Before each LLM call in a multi-step chain, the harness could inject a summary: "Steps completed: read main.rs, edited line 42. Remaining: run tests." This gives the model an external memory of its plan, compensating for its limited ability to track state across turns.

**Short chains should be the design target.** Rather than trying to make local models handle 10-step chains, Anima should optimize for 2-4 step interactions. The state machine (anima-states) is the right tool for longer workflows — it breaks a complex task into discrete states, each with a focused prompt, so the model only needs to handle a short chain within each state.

## Error Recovery

### Tool Not Found

When the model calls a nonexistent tool, the error is formatted and fed back:

```
[Tool Error for nonexistent_tool]
Error: Tool 'nonexistent_tool' not found
```

In native mode, this is a proper tool result message. The model sees the error and (usually) tries a different tool. In tool-block mode, the `list_tools` meta-tool provides a recovery path — the model can call it to see what's actually available.

### Bad Arguments

In native mode, when the JSON arguments fail to parse, the harness sends back the parse error plus the expected parameter schema:

```
[Tool Error for edit_file]
Invalid JSON in arguments: expected ',' or '}' at line 1 column 45. Raw: {"path":"main.rs"...

Expected parameters:
{"path": "string", "old": "string", "new": "string"}
```

This is effective — the schema gives the model exactly what it needs to retry.

### Silent Failures in Tool-Block Mode (Resolved)

This was the biggest error recovery gap: the model would try to call a tool but the output wouldn't match the extraction regex, and the harness would treat it as a final answer. Now resolved — the `<tool>` tag format is unambiguous (no collision with markdown), and if a `<tool>` tag is detected but extraction fails, a format error is fed back to the LLM with the expected format so it can retry.

### Infinite Loops

The `max_iterations` cap (default 25) is the backstop for infinite loops. The budget nudge at 50% and 80% provides softer intervention. But local models can still waste most of their budget on repetitive calls before the nudge takes effect.

A more aggressive strategy: detect repetition explicitly. If the model calls the same tool with the same arguments twice in a row, inject a warning: `[You already called read_file("main.rs") — the result is in your context above. Use the information you have or try a different approach.]` This catches the most common loop pattern (re-reading files) without waiting for the budget nudge.

## Recommendations for Anima

Based on the observations above, concrete changes ordered by impact:

### 1. ~~Fix the tool-block silent failure~~ (Done)

Implemented: switched from fenced code block delimiters to `<tool>...</tool>` XML tags. If a `<tool>` tag is present but extraction fails, a format error is fed back to the LLM. This eliminates silent termination on malformed tool calls.

### 2. Earlier deduplication

Run `dedup_tool_results` after every tool execution, not just at 90% context fill. The scan is cheap (linear over messages) and keeping context clean from the start improves reasoning quality throughout the chain. The 90% threshold can remain as the trigger for hard trimming.

### 3. Continuous budget awareness

Replace the 50%/80% threshold nudges with a per-turn status line appended to every tool result:

```
[Tool budget: 7 of 10 remaining]
```

This is lightweight enough to include every turn without being directive. Keep the stronger nudge language for 80%+ as an additional push.

### 4. Repetition detection

Before executing a tool call, check whether the exact same call (same tool name, same arguments) was made in the last 3 iterations. If so, skip execution and inject:

```
[Duplicate tool call — read_file("main.rs") was already called.
The result is in your conversation above. Use the existing result or try a different approach.]
```

This prevents the most common loop pattern without relying on the budget nudge.

### 5. ~~tool-block format reinforcement~~ (Done)

Format instructions are now included in the tool listing injected via `<recall>` every turn, using recency bias. The `<tool>` tag format is documented inline with the available tools list.

### 6. Smarter tool injection for native mode

Native mode currently injects all allowed tools every turn. For models with limited tool-handling capacity, filter down to the most relevant tools per turn — the same keyword-matching logic used in tool-block mode, but applied to native tool specs. Start with all tools on the first turn, then narrow to tools related to the current task based on the conversation so far.

### 7. Shorter default chain length

Reduce `max_iterations` from 25 to 10 for local models. Most useful work completes in 3-5 tool calls. A max of 10 provides headroom for complex tasks without letting the model burn 20 iterations on loops. This can be a model-level config rather than a global change — cloud models can keep higher limits.

### 8. Proactive context summarization

When the conversation exceeds 50% of context, instead of waiting for 90% to dedup, inject a system-generated summary of completed actions:

```
[Context summary]
Files read: src/main.rs (150 lines), src/lib.rs (80 lines)
Files modified: src/main.rs (line 42: fixed null check)
Commands run: cargo test (3 passed, 1 failed)
```

This gives the model an external memory of what's happened, reducing the urge to re-read files or re-run commands. The summary is much smaller than the raw tool results it represents.

## What Not to Change

Some things in Anima's current design work well and shouldn't be touched:

**The two-mode architecture.** Having both native and tool-block mode is the right call. Native mode is better when it works; tool-block mode is a reliable fallback. The ability to choose per-model is important.

**Database-backed context in daemon mode.** Reconstructing context from SQLite on each iteration (rather than relying on in-memory history) is architecturally sound. It enables pause/resume, multi-client access, and prevents context drift. The overhead is negligible compared to LLM inference time.

**Keyword-based tool recall.** Only showing 3-5 tools per query in tool-block mode is a strength, not a limitation. Local models perform better with fewer choices. The `list_tools` escape hatch handles the edge case where keyword matching picks the wrong tools.

**Tail-preserving truncation.** Keeping the end of tool output rather than the beginning is correct for almost all tools. Shell output, compiler errors, and test results all have the most important information at the tail.

**The `[Tool Result for X]` / `[Tool Error for X]` prefix convention.** Simple, consistent, and gives the model enough context to track which result came from which call. No need for more elaborate formatting.
