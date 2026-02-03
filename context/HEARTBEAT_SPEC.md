# Heartbeat Spec

## Overview

Agents wake up periodically, read their `heartbeat.md`, think, and log output to a dedicated conversation. This enables proactive behavior without human prompting.

---

## Design Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Heartbeat prompt | **heartbeat.md file** | Per-agent, editable, like persona.md |
| 2 | Output destination | **heartbeat-\<agent\> conversation** | Persistent, viewable, provides context |
| 3 | Busy handling | **Queue until free** | Don't interrupt mid-thought |
| 4 | Interval config | **Per-agent config.toml** | Different agents, different rhythms |

---

## Config

```toml
# ~/.anima/agents/arya/config.toml
[heartbeat]
enabled = true
interval = "30m"  # Duration string: "30s", "5m", "1h", "2h30m"
```

Default: disabled (no `[heartbeat]` section = no heartbeat)

---

## Files

```
~/.anima/agents/<name>/
├── config.toml      # existing + [heartbeat] section
├── persona.md       # existing
├── heartbeat.md     # NEW: prompt for periodic wakeup
└── memory.db        # existing
```

### heartbeat.md Example

```markdown
# Heartbeat

Check on things. Be proactive but not annoying.

## Things to check
- Any conversations I should follow up on?
- Anything in my memories that needs attention?
- Any tasks I said I'd do?

## Guidelines
- If nothing needs attention, just say "All clear."
- Don't be verbose — short status updates only.
- If something important, @mention relevant parties.
```

---

## Conversation

- **Name:** `<agent>-heartbeat` (e.g., `heartbeat-arya`)
- **Created automatically** on first heartbeat if doesn't exist
- **Participants:** just the agent (solo conversation)
- **Message flow:** Agent's heartbeat output stored as messages from the agent
- **Context:** Agent sees last N messages from this conversation (like any other)

This gives the agent memory of previous heartbeats — it can see what it checked last time.

---

## Daemon Behavior

### On startup (if heartbeat enabled):
```
1. Parse interval from config
2. Spawn heartbeat timer task
3. Timer fires every <interval>
```

### On heartbeat tick:
```
1. Check if agent is currently thinking
   - If busy: queue heartbeat (set pending flag)
   - If free: proceed
2. Load heartbeat.md content
3. Get/create <agent>-heartbeat conversation
4. Fetch last N messages for context
5. Think with heartbeat.md as user message
6. Store response in <agent>-heartbeat conversation
7. Parse @mentions from response → notify if any
```

### On think completion (if heartbeat was queued):
```
1. Clear pending flag
2. Immediately trigger heartbeat
```

---

## Implementation

### Config Changes (config.rs)

```rust
#[derive(Debug, Deserialize, Default)]
pub struct HeartbeatConfig {
    pub enabled: bool,
    pub interval: Option<String>,  // "30m", "1h", etc.
}

#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    // ... existing fields ...
    #[serde(default)]
    pub heartbeat: HeartbeatConfig,
}
```

### Daemon Changes (daemon.rs)

```rust
struct DaemonState {
    // ... existing fields ...
    heartbeat_pending: AtomicBool,
    thinking_lock: Mutex<()>,  // or use existing mechanism
}

// New task spawned on daemon start
async fn heartbeat_loop(state: Arc<DaemonState>, interval: Duration) {
    let mut ticker = tokio::time::interval(interval);
    loop {
        ticker.tick().await;
        
        // Try to acquire thinking lock
        if let Ok(_guard) = state.thinking_lock.try_lock() {
            run_heartbeat(&state).await;
        } else {
            // Agent is busy, queue it
            state.heartbeat_pending.store(true, Ordering::SeqCst);
        }
    }
}

// Called after any think() completes
fn on_think_complete(state: &DaemonState) {
    if state.heartbeat_pending.swap(false, Ordering::SeqCst) {
        // Queued heartbeat, run it now
        tokio::spawn(run_heartbeat(state.clone()));
    }
}

async fn run_heartbeat(state: &DaemonState) {
    // 1. Load heartbeat.md
    let heartbeat_prompt = fs::read_to_string(&state.heartbeat_path).ok();
    let Some(prompt) = heartbeat_prompt else { return };
    
    // 2. Get/create conversation
    let conv_name = format!("heartbeat-{}", state.agent_name);
    let conv_id = state.conv_store
        .get_or_create_conversation(&conv_name, &[&state.agent_name])
        .await?;
    
    // 3. Store heartbeat prompt as "system" or internal trigger (not shown?)
    // Or just use it directly without storing...
    
    // 4. Think
    let response = state.agent.think(&prompt).await?;
    
    // 5. Store response
    state.conv_store.add_message(&conv_id, &state.agent_name, &response).await?;
    
    // 6. Handle @mentions (reuse existing logic)
    let mentions = parse_mentions(&response);
    for mention in mentions {
        notify_agent(&mention, &conv_id, message_id).await;
    }
}
```

### Duration Parsing

```rust
fn parse_duration(s: &str) -> Option<Duration> {
    // Parse "30s", "5m", "1h", "2h30m" etc.
    // Use humantime crate or hand-roll
}
```

---

## CLI Commands

No new commands needed — use existing:

```bash
anima chat view heartbeat-arya           # See heartbeat history
anima chat view heartbeat-arya --limit 5 # Last 5 heartbeats
```

Optional future addition:
```bash
anima heartbeat <agent>                  # Manually trigger heartbeat
anima heartbeat <agent> --status         # Show next scheduled, last run
```

---

## Edge Cases

1. **Agent stops while heartbeat pending** — Clear pending flag on stop, don't persist
2. **heartbeat.md missing** — Skip heartbeat silently (disabled)
3. **heartbeat.md empty** — Skip heartbeat silently
4. **Interval shorter than think time** — Queue at most one, don't stack
5. **@mention in heartbeat response** — Handle normally, notify other agents

---

## Phases

### Phase 1: Basic Heartbeat
- Config parsing (`[heartbeat]` section)
- Heartbeat timer loop in daemon
- heartbeat-\<agent\> conversation creation
- Basic heartbeat execution

### Phase 2: Busy Handling
- Thinking lock / busy detection
- Queue pending heartbeat
- Trigger on think completion

### Phase 3: Polish
- `anima heartbeat` manual trigger command
- Status/diagnostics
- Heartbeat metrics (last run, next scheduled)

---

## Resolved Questions

1. **Should heartbeat prompt be stored in conversation?** 
   - **Decision: No** — just store the agent's response output, keep history clean

2. **First heartbeat timing?**
   - **Decision: After first interval** — don't spam on restart
   - Future: consider a separate "startup" event for immediate-on-boot behavior

3. **Heartbeat during pause?**
   - **Decision: Yes** — pause affects @mention forwarding, not agent thinking
