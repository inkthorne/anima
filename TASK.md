# Task: Implement `anima chat stop <conv>` command

## Purpose

Unpause a conversation WITHOUT processing queued notifications or catchup items. This is a "clean exit" from a sideways conversation — clears the queue without triggering any agent actions.

## Behavior

1. Unpause the conversation (set `paused = false`, clear `paused_at_msg_id`)
2. Delete all pending notifications for that conversation
3. Do NOT process catchup items or send any agent notifications
4. Print confirmation message

## Changes Required

### 1. `src/conversation.rs` — Add new method

Add after `delete_pending_notification()` (around line 1000):

```rust
/// Clear all pending notifications for a specific conversation.
/// Used by stop command to abandon queued work without processing.
pub fn clear_pending_notifications_for_conversation(&self, conv_name: &str) -> Result<usize, ConversationError> {
    let deleted = self.conn.execute(
        "DELETE FROM pending_notifications WHERE conv_name = ?1",
        params![conv_name],
    )?;
    Ok(deleted)
}
```

### 2. `src/main.rs` — Add CLI subcommand

Add to `ChatCommands` enum (after Resume, around line 265):

```rust
/// Stop a paused conversation without processing queued notifications. Supports glob patterns.
Stop {
    /// Name or pattern of the conversation(s) to stop
    conv: String,
    /// Skip confirmation for multiple matches
    #[clap(short, long)]
    force: bool,
},
```

### 3. `src/main.rs` — Add handler

Add after the Resume handler (around line 1665):

```rust
// `anima chat stop <conv>` - stop a paused conversation without catchup
Some(ChatCommands::Stop { conv, force }) => {
    let matches = store.match_conversations(&conv)?;

    if matches.is_empty() {
        return Err(format!("No conversations match pattern: {}", conv).into());
    }

    if !force && (matches.len() > 1 || has_wildcards(&conv)) {
        if !confirm_action("stop", &matches)? {
            println!("Aborted.");
            return Ok(());
        }
    }

    for conv_name in &matches {
        // Unpause without catchup processing
        let _ = store.set_paused(conv_name, false)?;
        
        // Clear pending notifications (don't process them)
        let cleared = store.clear_pending_notifications_for_conversation(conv_name)?;
        
        if cleared > 0 {
            println!("Stopped '{}' ({} pending notification(s) cleared)", conv_name, cleared);
        } else {
            println!("Stopped '{}'", conv_name);
        }
    }
}
```

## Build & Test

```bash
cd ~/dev/anima
cargo build
cargo test
```

Both must pass before requesting review.

## Workflow

1. @dash implements the changes
2. @check reviews when dash is ready
3. Check approves with "LGTM ✓" (no @mention) when satisfied
