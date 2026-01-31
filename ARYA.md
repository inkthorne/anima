# ARYA.md — Timer Triggers Implementation

## Goal
Add periodic timer triggers so agents can have heartbeats — wake up on a schedule to think.

## Implementation Plan

### Phase 1: Duration Parsing
- [ ] Add `parse_duration()` function to handle formats: `30s`, `5m`, `1h`, `1h30m`
- [ ] Location: `src/util.rs` or new `src/duration.rs`

### Phase 2: Extend Agent Config
- [ ] Add `timer_interval: Option<Duration>` to AgentConfig
- [ ] Add `timer_message: Option<String>` for custom trigger text
- [ ] Update config serialization

### Phase 3: CLI Flags
- [ ] Add `--every <duration>` flag to `agent start`
- [ ] Add `--on-timer <message>` flag (optional, defaults to "Timer wake")
- [ ] Parse and validate duration at command line

### Phase 4: Timer Loop Integration
- [ ] Modify agent run loop to use `tokio::select!` with:
  - Inbox message receiver
  - Timer interval (if configured)
- [ ] On timer tick, inject timer message into agent's processing
- [ ] Timer should NOT fire if agent is already processing

### Phase 5: Timer Status Command
- [ ] Add `timer status` subcommand to show active timers
- [ ] Display: agent name, interval, last fired, next fire time

## Design Notes
- Timer is per-agent, not global
- Timer stops when agent stops
- If agent is busy when timer fires, skip that tick (don't queue)
- Duration stored in agent state file for persistence across restarts

## Example Commands
```bash
agent start alice --every 5m
agent start alice --every 30s --on-timer "Check for updates"
timer status
agent stop alice
```
