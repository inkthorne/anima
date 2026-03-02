## Overview

A standalone TDD coding agent built on anima-states. The user gives it a coding request and it walks through the red-green-refactor cycle: plan the work, write tests, write implementation, run the tests, fix failures, refactor, and confirm. It has tool access for file I/O and shell execution.

The agent lives at `~/anima/agents/tdd/` with the standard structure: `config.toml`, `system.md`, and a `states/` directory containing eight state files.

This spec covers the state graph, variable flow, and prompt sketches. Agent-level configuration (`config.toml` settings, `system.md` content, tool definitions) is outside its scope — the agent is assumed to be configured with file I/O and shell tools.

## States

Eight states forming this graph:

```
initial → plan → tests → implement → verify
  ↑  ↺                                ↓  ↑
  │                                   fix─┘
  │                                    ↓
  │                                 refactor → verify → done
  └──────────────────────────────────────────────────────┘
```

### initial

Receive user input. Determine whether it's a coding request. If yes, move to plan. If not, respond that this agent only handles coding tasks and reset to initial (ready for the next input). Pure reasoning — no tools.

**Sets (→ plan):** (none)
**Sets (→ initial):** (none)

### plan

Analyze the user's request. Produce structured output: requirements, components to build, edge cases to cover, and file paths. Uses tools to read any files or specs referenced in the user's message — does not write anything.

The plan must include: the programming language, test framework, test command, target file paths, and any project scaffolding commands needed (e.g., `cargo init`, `npm init`).

**Sets:** `plan`

### tests

Write comprehensive test cases based on the plan. Use the test framework and conventions specified in the plan. Cover happy paths, edge cases, and error conditions. Pure reasoning — no tools.

**Variables in:** `plan`
**Sets:** `tests`

### implement

Write implementation code intended to pass the tests. Pure reasoning — no tools.

**Variables in:** `plan`, `tests`
**Sets:** `implementation`, `phase`, `loop_count`

### verify

The only state that writes to disk. Scaffold the project if needed (run init commands from the plan), write files to disk, run the test suite, assess results. Decides next step based on outcome:

- Tests fail → **fix**
- Tests pass (first time, pre-refactor) → **refactor**
- Tests pass (post-refactor) → **done**

Verify reads `{{phase}}` to decide the pass transition: `pre-refactor` means go to refactor on pass, `post-refactor` means go to done on pass.

**Variables in:** `plan`, `tests`, `implementation`, `phase`, `loop_count`
**Sets (→ fix):** `test_output`
**Sets (→ refactor):** (none)
**Sets (→ done):** (none)

### fix

Analyze test failures from verify's output. Produce corrected implementation. Pure reasoning — no tools.

**Variables in:** `plan`, `tests`, `implementation`, `test_output`, `phase`, `loop_count`
**Sets (→ verify):** `implementation` (updated), `loop_count`
**Sets (→ done):** (none)

### refactor

Clean up the passing implementation: naming, structure, duplication removal. Must not change behavior. Pure reasoning — no tools.

**Variables in:** `plan`, `tests`, `implementation`
**Sets:** `implementation` (updated), `phase`, `loop_count`

### done

Present a summary to the user: what was built, what's tested, final file listing. Emits `<clear-vars />` and transitions back to initial, ready for the next request.

**Variables in:** `tests`, `implementation`
**Clears:** all variables via `<clear-vars />`

## Variable Flow

Variables persist for the entire session (per anima-states). Each state only emits variables it introduces or modifies — earlier values remain available automatically. The table below shows what each transition *sets or updates*, not the full set of variables available:

| From | To | Sets / Updates |
|---|---|---|
| initial | plan | (none) |
| initial | initial | (none) |
| plan | tests | `plan` |
| tests | implement | `tests` |
| implement | verify | `implementation`, `phase`, `loop_count` |
| verify | fix | `test_output` |
| verify | refactor | (none) |
| verify | done | (none) |
| fix | verify | `implementation`, `loop_count` |
| fix | done | (none) |
| refactor | verify | `implementation`, `phase`, `loop_count` |
| done | initial | `<clear-vars />` |

Variables accumulate as the session progresses: `plan` is set once in the plan state and remains available through verify, fix, and refactor without re-emission. The `done` state emits `<clear-vars />` to reset the variable map before returning to initial, ensuring the next coding request starts with a clean slate.

## Prompt Sketches

What each state's `.md` file should contain.

### states/initial.md

```
---
wait: true
---
```

**Goal:** Determine whether the user's input is a coding request.

**Rules:**
- If the input is a coding task (build something, fix a bug, write a function, etc.), proceed to plan.
- If the input is not a coding task, respond that this agent only handles coding requests and reset to initial.

**Output format:** A brief acknowledgment (if coding) or a polite decline (if not).

**Transitions:**
- `plan` — input is a coding request.
- `initial` — input is not a coding request.

### states/plan.md

**Goal:** Break the user's request into a structured plan.

**Rules:**
- If the user references files or specs, read them first using tools.
- Do not write code or tests yet.
- Output a numbered list of requirements, a list of components/modules, edge cases, and target file paths.
- Include: programming language, test framework, test command, and any project scaffolding commands needed (e.g., `cargo init`, `npm init`).
- Be specific enough that someone else could write the tests from this plan alone.

**Output format:** Freeform markdown under clear headings (Requirements, Components, Edge Cases, Files, Language & Tools).

**Transitions:** Always → `tests`.

### states/tests.md

**Goal:** Write a complete test suite from the plan.

**Rules:**
- Use the test framework and conventions specified in the plan.
- Cover every requirement and edge case from the plan.
- Include at least one test per happy path, one per edge case, one per error condition.
- Do not write implementation code.

**Input:** `{{plan}}`

**Output format:** A single fenced code block labeled with its file path, containing the full test file.

**Transitions:** Always → `implement`.

### states/implement.md

**Goal:** Write implementation code that passes the tests.

**Rules:**
- Write the minimum code to make all tests pass.
- Follow the file paths from the plan.
- Do not modify the tests.
- Set `phase` to `pre-refactor`.
- Set `loop_count` to `0`.

**Input:** `{{plan}}`, `{{tests}}`

**Output format:** Fenced code blocks, one per file, each labeled with its file path.

**Transitions:** Always → `verify`.

### states/verify.md

**Goal:** Scaffold project if needed, write files to disk, run tests, assess results.

**Rules:**
- If the plan specifies scaffolding commands (e.g., `cargo init`), run them first.
- Use tools to write `{{tests}}` and `{{implementation}}` to the file paths established in the plan.
- Run the test command specified in the plan.
- Report the full test output.
- Decide the next state based on results and entry point.

**Input:** `{{plan}}`, `{{tests}}`, `{{implementation}}`, `{{phase}}`, `{{loop_count}}`

**Output format:** Test output followed by a pass/fail assessment.

**Transitions:**
- `fix` — one or more tests failed.
- `refactor` — all tests passed and `{{phase}}` is `pre-refactor`.
- `done` — all tests passed and `{{phase}}` is `post-refactor`.

### states/fix.md

**Goal:** Diagnose test failures and produce corrected implementation.

**Rules:**
- Read the test output carefully. Identify root causes.
- Fix only the implementation, not the tests (unless a test has an obvious bug like a typo).
- Explain what went wrong and what you changed.
- Increment `loop_count` by 1. If `{{loop_count}}` reaches 3, stop attempting fixes. Transition to done with a summary of what failed and why.

**Input:** `{{plan}}`, `{{tests}}`, `{{implementation}}`, `{{test_output}}`, `{{phase}}`, `{{loop_count}}`

**Output format:** Diagnosis, then corrected fenced code blocks.

**Transitions:**
- `verify` — `loop_count` < 3.
- `done` — `loop_count` reached 3. Include a summary of what failed and why.

### states/refactor.md

**Goal:** Improve code quality without changing behavior.

**Rules:**
- Improve naming, reduce duplication, simplify structure.
- Do not add features or change the public interface.
- Do not modify tests.
- If the code is already clean, make no changes and say so.
- Set `phase` to `post-refactor`.
- Set `loop_count` to `0`.

**Input:** `{{plan}}`, `{{tests}}`, `{{implementation}}`

**Output format:** Refactored fenced code blocks (or a note that no changes were needed).

**Transitions:** Always → `verify`.

### states/done.md

**Goal:** Summarize the completed work.

**Rules:**
- List what was built, what's tested, and the final files on disk.
- Keep it concise.
- Emit `<clear-vars />` before transitioning to clear the session's variable map.

**Input:** `{{tests}}`, `{{implementation}}`

**Output format:** Summary with file listing.

**Transitions:** Always → `initial` (with `<clear-vars />`).

## Transition Graph

```
  ┌───────────────────────────────────────────────────────────┐
  │                                                           │
  │  ┌──────────────────────────────────────────┐             │
  │  │                                          │             │
  ▼  ↺                                          │             │
initial ──→ plan ──→ tests ──→ implement ──→ verify ────┤     │
                                              │  ▲      │     │
                                      fail    │  │      │     │
                                              ▼  │      │     │
                                             fix─┘      │     │
                                                        │     │
                                              pass      │     │
                                          (1st time)    │     │
                                              │         │     │
                                              ▼         │     │
                                           refactor ──→ verify│
                                                          │   │
                                                          │   │
                                                    pass  │   │
                                             (post-refactor)  │
                                                          │   │
                                                          ▼   │
                                                        done──┘
```

Every state has at least one outgoing transition. Every state is reachable from initial. The initial↺initial self-loop handles non-coding input. The verify↔fix loop can cycle until tests pass or `loop_count` reaches 3, at which point fix bails out to done with a failure summary. The done→initial transition makes the agent ready for the next request.
