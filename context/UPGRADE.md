# Package Upgrade Plan

Current versions are pinned for stability. Upgrade one-at-a-time to catch breaking changes.

## Packages to Upgrade

| Package | Current | Latest | Breaking? | Notes |
|---------|---------|--------|-----------|-------|
| dirs | 5.0 | 6.0 | Minor | API mostly same |
| reqwest | 0.12 | 0.13 | Yes | `rustls-tls` → `rustls` feature |
| rusqlite | 0.32 | 0.38 | **Yes** | `u64` no longer impl `ToSql/FromSql` |
| rustyline | 14 | 17 | Maybe | Check API changes |
| thiserror | 1 | 2 | Maybe | Check derive changes |
| toml | 0.8 | 0.9 | Minor | Likely compatible |

## Upgrade Order (safest first)

### 1. toml 0.8 → 0.9
```bash
# Edit Cargo.toml: toml = "0.9"
cargo build && cargo test
```

### 2. dirs 5 → 6
```bash
# Edit Cargo.toml: dirs = "6"
cargo build && cargo test
```

### 3. thiserror 1 → 2
```bash
# Edit Cargo.toml: thiserror = "2"
cargo build && cargo test
# May need to update error derive syntax
```

### 4. rustyline 14 → 17
```bash
# Edit Cargo.toml: rustyline = "17"
cargo build && cargo test
# Check REPL still works interactively
```

### 5. reqwest 0.12 → 0.13
```bash
# Edit Cargo.toml: reqwest = { version = "0.13", features = ["json", "rustls", "stream"] }
# Note: feature renamed from "rustls-tls" to "rustls"
cargo build && cargo test
```

### 6. rusqlite 0.32 → 0.38 (most work)
```bash
# Edit Cargo.toml: rusqlite = { version = "0.38", features = ["bundled"] }
```

**Required code changes in `src/memory.rs`:**
- Change `u64` timestamps to `i64` in structs
- Update `now()` function to return `i64`
- Cast appropriately in SQL params
- Update `FromSql` reads to cast `i64` → `u64` if needed

Files affected:
- `src/memory.rs` (MemoryEntry struct, queries)

## After Each Upgrade

1. `cargo build` — check compilation
2. `cargo test` — run all tests
3. Manual test: `anima ask arya "test"`
4. Commit if passing

## Rust Version

Some packages require newer Rust:
- `home v0.5.12` requires Rust 1.88
- `wasip2 v1.0.2` requires Rust 1.87

Update Rust with: `rustup update`
