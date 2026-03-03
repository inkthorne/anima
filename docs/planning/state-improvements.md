# Anima States — Competitive Position & Fragility Mitigations

## Competitive Analysis Summary

Anima states occupies a genuinely novel niche that no other framework serves:

| | Anima States | LangGraph | CrewAI Flows | AutoGen | Semantic Kernel |
|---|---|---|---|---|---|
| **Definition** | Markdown files | Python graph API | Python decorators | Python functions | Class-based + events |
| **Transitions** | LLM-emitted tags | Code (conditional edges) | Code (@router) | Code (speaker selection) | Code (event routing) |
| **State passing** | `<set-vars>` XML tags | TypedDict / Pydantic | Pydantic + method args | Conversation history | Pydantic state models |
| **Target user** | Prompt writers | Python developers | AI engineers | Python developers | Enterprise .NET/Python |
| **Requires code** | No | Yes | Yes | Yes | Yes |

**Unique strengths:**
1. **Markdown-as-workflow** — no code required, states are just prompt files
2. **LLM-navigated** — the agent chooses its own path, not code
3. **Daemon-integrated** — workflows live inside persistent agents, not ephemeral runs
4. **Accessible** — anyone who writes prompts can define agent behavior

**Not reinventing the wheel.** The niche is "structured agent behavior without writing code." Nobody else occupies it.

---

## Fragility Concerns & Mitigations

### Concern 1: Invalid state transition loops

**Problem:** LLM hallucinating state names -> error feedback -> LLM tries again -> no max retry -> infinite loop burning tokens.

**Current guardrail:** Error message with available states list (daemon.rs).

**Mitigation: Max retry counter** (implemented)

Added a counter for consecutive invalid state transitions. After 3 failures, keeps the current state and returns the response with a warning.

**File:** `src/daemon.rs` — state transition error handling

- `state_error_count: u32` initialized near loop start
- On invalid state: increment counter
- If counter >= 3: log warning, keep current state, return response with warning appended
- On valid transition: reset counter to 0

### Concern 2: LLM burden on linear transitions

**Problem:** For workflows where transitions are always the same (plan -> tests -> implement), the LLM still has to remember to emit `<set-vars><state>tests</state></set-vars>`. This is unnecessary friction and a failure point.

**Mitigation: `default_next` frontmatter key** (implemented)

States can declare a default next state:

```markdown
---
wait: false
default_next: tests
---
Write tests based on the plan...
```

If the LLM doesn't emit a `<state>` tag, the harness uses `default_next` instead of keeping the current state. If `default_next` is not set, behavior is unchanged (existing v4.4.8 behavior preserved).

**Files:**
- `src/pipeline.rs` — `default_next: Option<String>` added to `StateFrontmatter`, parsed in `parse_state_frontmatter()`
- `src/daemon.rs` — in the "no state change" branch, checks frontmatter for `default_next`, validates the target state file exists, follows the same wait/non-wait logic as explicit transitions

### Concern 3: Context window pressure

**Problem:** Full conversation history on every turn. A TDD agent going through plan -> tests -> implement -> verify -> fix -> verify accumulates massive context, especially with large `{{implementation}}` values template-substituted AND in history.

**Assessment:** This is real but premature to engineer around. The TDD agent's 8 states are manageable. Context pressure becomes a problem with 20+ state transitions or very large variable values.

**Recommendation: Defer.** Document as a known design consideration. Revisit if/when agents hit context limits in practice. Future options include:
- Per-state `history_limit` frontmatter to truncate visible history
- Auto-summarization of older state turns
- Variable-aware deduplication (don't re-inject via template what's already in history)

### Concern 4: Untyped variables / `<set-vars>` reliability

**Problem:** No validation that required variables are set. Missing vars left as literal `{{name}}` text.

**Assessment:** Current behavior is actually informative — the LLM sees the literal `{{plan}}` and knows something went wrong. Adding `requires: [plan, tests]` in frontmatter would catch this earlier, but the recovery path is unclear (can't go back and re-run the previous state).

**Recommendation: Defer.** The literal-preservation behavior is a reasonable fallback. The LLM self-corrects when it sees unresolved template variables. Revisit only if this causes real problems in practice.

---

## Implementation Summary

### Done

1. **Max retry on invalid state transitions** — `src/daemon.rs`
2. **`default_next` frontmatter key** — `src/pipeline.rs`, `src/daemon.rs`
3. **Tests** — 5 new tests in `src/pipeline.rs` for frontmatter parsing

### Deferred

4. Context pressure mitigation — wait for real-world evidence
5. Required variable validation — current literal-preservation is sufficient
6. Variable size limits — not a practical problem yet
7. Transition restriction lists — file-exists check is sufficient
