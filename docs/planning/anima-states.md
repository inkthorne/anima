## Goal

States are an optional feature that allows the user to define a sequence of discrete steps that guide the LLM through a task. Rather than giving the LLM a single monolithic prompt, the user breaks their intent into a state machine where each state carries its own focused instructions. The LLM executes one state at a time and transitions to the next based on its output.

The harness does not require states to function. State behavior is only active when the agent's `config.toml` specifies an initial state. Without it, the harness operates as a standard conversation loop.

For example, a user could define a test-driven development workflow as a series of states: first a **write-tests** state that instructs the LLM to produce tests for a given specification, then an **implement** state that instructs it to write code that passes those tests.

## Agent Structure

An agent is a directory at `~/anima/agents/<agent name>/` containing:

- `config.toml` - Agent configuration, including the initial state and model settings.
- `system.md` - The system prompt, sent as the system message on every turn regardless of the current state.
- `states/` - Directory containing the state files.

This spec defines the state machine — how states are loaded, transitioned, and how variables flow between them. The contents of `config.toml` (model settings, tool definitions, etc.) and `system.md` are outside this spec's scope. The state machine assumes the agent is already configured with whatever tools and system prompt it needs.

## State Files

A state is a Markdown file (`.md`) in the agent's `states/` directory that serves as the user message for a single turn in the conversation. The file contains the prompt text along with template variables that are replaced with dynamic values at runtime. Template variables use double-brace syntax: `{{variable}}`.

State files may begin with optional YAML frontmatter delimited by `---` fences. The harness strips frontmatter before sending the state's content to the LLM. The only frontmatter property currently defined is `wait`:

```markdown
---
wait: true
---

Your prompt text here...
```

- `wait: true` means the harness pauses and waits for the user to send a message before running this state. The user's message becomes `{{user}}`.
- When any state transitions to a wait state, the current chain ends. The harness does not send a new turn to the LLM until the user provides input.
- The initial state (as specified in `config.toml`) should have `wait: true` — it's the entry point that receives the user's request.
- Non-wait states auto-generate their user message from the template and run immediately.

The following template variables are available:

- `{{user}}` — The most recent message the user typed. This value is set when a `wait` state receives user input and remains the same for all subsequent non-wait states in the chain.
- `{{assistant}}` - The LLM's output from the previous turn.

The LLM sees the full conversation history on every turn, including all prior state prompts and responses. This allows each state to build on context established by previous states in the chain.

Each state file should list its available transitions as part of the prompt, so the LLM knows which states it can move to and under what conditions. A recommended convention is to include a `## Transitions` section at the end of the file:

```markdown
<!-- categorize.md -->

Classify the user's message into one of the following categories
and transition to the appropriate state.

{{user}}

## Transitions
- greeting: The message is a greeting or salutation.
- question: The message is asking for information.
- action: The message is requesting the agent to do something.
- silence: The message is not directed at the agent.
```

The descriptions serve double duty: they tell the LLM what the available transitions are and when to choose each one.

## State Transitions

The initial state is specified in the agent's `config.toml` file. On each subsequent turn, the LLM controls which state runs next by emitting a `<next-state>` XML tag in its output. The harness parses this tag, appends `.md`, and loads the corresponding file as the next turn's user message. The `<next-state>` tag is kept in the conversation history so the LLM can see the pattern of its own prior transitions, which helps it remember to emit the tag consistently.

The LLM is **required** to emit a `<next-state>` tag on every non-tool-calling turn when states are active (see **Tool Use** below for the full rule). To remain in the current state, the LLM emits the current state's name as the value (e.g. `<next-state>initial</next-state>` while in the initial state). If the LLM fails to emit a `<next-state>` tag, or emits a state name that does not correspond to an existing file, the harness notifies the LLM of the error and prompts it to choose again.

For example:

```
<next-state>categorize</next-state>
```

## Tool Use

When the LLM's response includes tool calls, the harness stays in the current state. It executes the tools, feeds the results back to the LLM, and lets it respond again. This loop repeats until the LLM produces a response with no tool calls. The harness only inspects this final response for `<next-state>` and `<set-vars>` tags — intermediate tool-calling responses are invisible to the state machine.

From the state machine's perspective, the entire tool loop collapses into a single turn. A state that calls three tools before producing its answer counts as one turn with one transition, not four.

The `{{assistant}}` template variable resolves to the LLM's final text response from the previous turn — the one with no tool calls. Tool call requests and tool results are not included. Agents that need tool output in subsequent states should pass it explicitly via `<set-vars>`.

## Variables

States can pass arbitrary values to later states using the `<set-vars>` XML tag. The LLM emits this tag in its output with each variable as a child element:

```
<set-vars>
  <color>yellow</color>
  <size>large</size>
</set-vars>
```

Variable values may contain any text, including characters that are meaningful in XML (`<`, `>`, `&`). The harness does not use a strict XML parser. It locates each child element's opening and closing tags by name (e.g. `<color>` … `</color>`) and treats everything between them as a literal string. This means variable values can contain code, angle brackets, and other markup without escaping. The only restriction is that a variable's value cannot contain its own closing tag (e.g. the value of `<color>` cannot contain the literal string `</color>`).

Later states can reference these values using template syntax: `{{color}}`, `{{size}}`, etc.

Variables persist for the entire session. Each `<set-vars>` block merges into the session's variable map — new keys are added and existing keys are overwritten with the new value. A variable set in state A remains available in states B, C, D, etc. without being re-emitted. States only need to emit variables they introduce or modify.

To reset the variable map (e.g. at session boundaries), the LLM emits a `<clear-vars />` self-closing tag. This empties the entire variable map. If both `<clear-vars />` and `<set-vars>` appear in the same turn, `<clear-vars />` is processed first, then `<set-vars>` — allowing a state to clear everything and set fresh values in one response.

```
<clear-vars />
<next-state>initial</next-state>
```

`<clear-vars />` is stripped from conversation history, the same as `<set-vars>`.

If a template references a variable that has not been set during the session and is not a built-in (`{{user}}`, `{{assistant}}`), the harness leaves the literal `{{name}}` in the text unchanged. This makes missing variables visible in the LLM's input, aiding debugging.

## LLM Output

A complete LLM response during state execution contains the response content, optional variables, and the required state transition:

```
Based on your message, I'll help you with that question.

Here's what I found: the capital of France is Paris.

<set-vars>
  <category>question</category>
  <topic>geography</topic>
</set-vars>
<next-state>final</next-state>
```

The harness strips `<set-vars>` and `<next-state>` from the output before displaying it to the user.

`<next-state>` is kept in the conversation history (as noted in **State Transitions**). `<set-vars>` is stripped from history — the variable values are already substituted into the next state's template, so they remain visible in context without duplicating potentially large content.

## Example

Consider a conversational agent that monitors a chat and only responds when addressed. The user defines the following states:

1. **initial** - The LLM scans the chat history and determines whether any new messages are directed at the agent.
2. **silence** - If no messages are for the agent, it produces no output and transitions back to **initial**.
3. **categorize** - If a message is for the agent, the LLM classifies it into a category: greeting, question, opinion, action, etc.
4. **greeting**, **question**, etc. - Each category has its own state with tailored instructions for how the agent should respond.
5. **final** - The LLM outputs its final message to the user and transitions back to **initial**, ready for the next interaction.

The transitions form a graph: **initial** branches to either **silence** or **categorize**, **categorize** branches to the appropriate response state based on the classification, and each response state transitions to **final** when its work is done.
