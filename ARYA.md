# ARYA.md — Current Task

## v2.5: Channel Integrations

Connect anima agents to messaging platforms so they can talk to the world.

### Goal

- Telegram bot integration (first priority)
- Message routing to specific agents
- Generic adapter pattern for future channels (Discord, webhooks, etc.)

### Implementation Plan

#### Phase 1: Telegram Bot Foundation
- [ ] Add `teloxide` crate for Telegram bot API
- [ ] Create `src/channels/mod.rs` module
- [ ] Create `src/channels/telegram.rs` with basic bot setup
- [ ] Bot can receive messages and echo them back

#### Phase 2: Agent Integration
- [ ] Route incoming Telegram messages to a designated agent
- [ ] Agent responses sent back to Telegram chat
- [ ] Support for configuring which agent handles Telegram

#### Phase 3: Multi-User Support
- [ ] Track chat IDs for different conversations
- [ ] Conversation history per chat
- [ ] Optional: spawn separate agent per user

#### Phase 4: Configuration
- [ ] Add Telegram config to `config.toml`
- [ ] Bot token from env var or config
- [ ] Agent name mapping in config

#### Phase 5: Graceful Lifecycle
- [ ] Clean shutdown of Telegram polling
- [ ] Reconnection on network errors
- [ ] Status command to show connected channels

### Design Notes

- Channel is a separate concern from Agent — agents don't know about Telegram
- Channel adapters translate platform messages → agent inbox, agent responses → platform
- One Telegram bot can route to multiple agents (by command or config)
- Start with long polling, upgrade to webhooks later if needed

### Example Config

```toml
[channels.telegram]
enabled = true
token = "${TELEGRAM_BOT_TOKEN}"
default_agent = "arya"
```

### Example Flow

```
User (Telegram) → "Hello!"
  ↓
TelegramChannel receives message
  ↓
Routes to agent "arya" inbox
  ↓
Agent thinks, responds: "Hi there!"
  ↓
TelegramChannel sends response
  ↓
User sees reply in Telegram
```

### Success Criteria

- [ ] Can run `cargo run` and have a working Telegram bot
- [ ] Messages in Telegram trigger agent thinking
- [ ] Agent responses appear in Telegram
- [ ] Clean shutdown with Ctrl+C
