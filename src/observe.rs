//! Observability infrastructure for the Anima runtime.
//!
//! Provides event-based monitoring, logging, and metrics collection.

use async_trait::async_trait;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Events that can be observed during agent execution.
#[derive(Debug, Clone, Serialize)]
pub enum Event {
    /// Agent started processing a task
    AgentStart {
        agent_id: String,
        task: String,
    },
    /// Agent completed a task
    AgentComplete {
        agent_id: String,
        duration_ms: u64,
        success: bool,
    },
    /// LLM API call completed
    LlmCall {
        model: String,
        tokens_in: Option<u32>,
        tokens_out: Option<u32>,
        duration_ms: u64,
    },
    /// Tool was invoked
    ToolCall {
        tool_name: String,
        duration_ms: u64,
        success: bool,
        error: Option<String>,
        /// Tool input parameters for logging context
        params: Option<serde_json::Value>,
        /// Brief summary of result (e.g., "path=/foo/bar.txt bytes=1234")
        result_summary: Option<String>,
    },
    /// Retry attempt for an operation
    Retry {
        operation: String,
        attempt: usize,
        delay_ms: u64,
    },
    /// Error occurred
    Error {
        context: String,
        message: String,
    },
}

/// Trait for observability backends.
///
/// Implement this trait to create custom observation handlers
/// (e.g., file logging, metrics exporters, distributed tracing).
#[async_trait]
pub trait Observer: Send + Sync {
    /// Called when an event occurs.
    async fn observe(&self, event: Event);
}

/// Console-based observer that prints events to stderr.
pub struct ConsoleObserver {
    /// When true, prints all events. When false, only errors and completions.
    pub verbose: bool,
}

impl ConsoleObserver {
    /// Create a new console observer.
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

#[async_trait]
impl Observer for ConsoleObserver {
    async fn observe(&self, event: Event) {
        match &event {
            Event::AgentStart { agent_id, task } => {
                if self.verbose {
                    eprintln!("[AGENT] {} starting: {}", agent_id, truncate(task, 50));
                }
            }
            Event::AgentComplete { agent_id, duration_ms, success } => {
                if self.verbose || !success {
                    let status = if *success { "completed" } else { "failed" };
                    eprintln!("[AGENT] {} {} in {}ms", agent_id, status, duration_ms);
                }
            }
            Event::LlmCall { model, tokens_in, tokens_out, duration_ms } => {
                if self.verbose {
                    let tokens = match (tokens_in, tokens_out) {
                        (Some(i), Some(o)) => format!(" ({}→{} tokens)", i, o),
                        _ => String::new(),
                    };
                    eprintln!("[LLM] {} call took {}ms{}", model, duration_ms, tokens);
                }
            }
            Event::ToolCall { tool_name, duration_ms, success, error, result_summary, .. } => {
                if self.verbose || !success {
                    let status = if *success {
                        "ok".to_string()
                    } else {
                        format!("error: {}", error.as_deref().unwrap_or("unknown"))
                    };
                    let summary = result_summary.as_deref().map(|s| format!(" {}", s)).unwrap_or_default();
                    eprintln!("[TOOL] {} {}{} ({}ms)", tool_name, status, summary, duration_ms);
                }
            }
            Event::Retry { operation, attempt, delay_ms } => {
                if self.verbose {
                    eprintln!("[RETRY] {} attempt {} after {}ms", operation, attempt, delay_ms);
                }
            }
            Event::Error { context, message } => {
                eprintln!("[ERROR] {}: {}", context, message);
            }
        }
    }
}

/// Collects metrics in memory for later inspection.
pub struct MetricsCollector {
    /// Total number of LLM calls made
    pub llm_calls: AtomicU64,
    /// Total number of tool calls made
    pub tool_calls: AtomicU64,
    /// Total number of errors encountered
    pub errors: AtomicU64,
    /// Total tokens consumed (input + output)
    pub total_tokens: AtomicU64,
    /// Total retry attempts
    pub retries: AtomicU64,
    /// Total time spent in LLM calls (milliseconds)
    pub llm_time_ms: AtomicU64,
    /// Total time spent in tool calls (milliseconds)
    pub tool_time_ms: AtomicU64,
}

impl MetricsCollector {
    /// Create a new metrics collector with all counters at zero.
    pub fn new() -> Self {
        Self {
            llm_calls: AtomicU64::new(0),
            tool_calls: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            retries: AtomicU64::new(0),
            llm_time_ms: AtomicU64::new(0),
            tool_time_ms: AtomicU64::new(0),
        }
    }

    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            llm_calls: self.llm_calls.load(Ordering::Relaxed),
            tool_calls: self.tool_calls.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            total_tokens: self.total_tokens.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            llm_time_ms: self.llm_time_ms.load(Ordering::Relaxed),
            tool_time_ms: self.tool_time_ms.load(Ordering::Relaxed),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Observer for MetricsCollector {
    async fn observe(&self, event: Event) {
        match event {
            Event::LlmCall { tokens_in, tokens_out, duration_ms, .. } => {
                self.llm_calls.fetch_add(1, Ordering::Relaxed);
                self.llm_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
                if let Some(t_in) = tokens_in {
                    self.total_tokens.fetch_add(t_in as u64, Ordering::Relaxed);
                }
                if let Some(t_out) = tokens_out {
                    self.total_tokens.fetch_add(t_out as u64, Ordering::Relaxed);
                }
            }
            Event::ToolCall { duration_ms, success, .. } => {
                self.tool_calls.fetch_add(1, Ordering::Relaxed);
                self.tool_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
                if !success {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            Event::Retry { .. } => {
                self.retries.fetch_add(1, Ordering::Relaxed);
            }
            Event::Error { .. } => {
                self.errors.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

/// A point-in-time snapshot of collected metrics.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub llm_calls: u64,
    pub tool_calls: u64,
    pub errors: u64,
    pub total_tokens: u64,
    pub retries: u64,
    pub llm_time_ms: u64,
    pub tool_time_ms: u64,
}

/// Observer that logs events using an AgentLogger.
///
/// This observer integrates the observability system with the AgentLogger,
/// providing unified logging with timestamps and agent prefixes.
pub struct AgentLoggerObserver {
    logger: std::sync::Arc<crate::daemon::AgentLogger>,
}

impl AgentLoggerObserver {
    /// Create a new observer that logs to the given AgentLogger.
    pub fn new(logger: std::sync::Arc<crate::daemon::AgentLogger>) -> Self {
        Self { logger }
    }
}

#[async_trait]
impl Observer for AgentLoggerObserver {
    async fn observe(&self, event: Event) {
        match &event {
            Event::AgentStart { agent_id: _, task } => {
                self.logger.log(&format!("Agent starting: {}", truncate(task, 80)));
            }
            Event::AgentComplete { agent_id: _, duration_ms, success } => {
                let status = if *success { "completed" } else { "failed" };
                self.logger.log(&format!("Agent {} in {}ms", status, duration_ms));
            }
            Event::LlmCall { model, tokens_in, tokens_out, duration_ms } => {
                let tokens = match (tokens_in, tokens_out) {
                    (Some(i), Some(o)) => format!(" ({}→{} tokens)", i, o),
                    _ => String::new(),
                };
                self.logger.log(&format!("LLM {} call took {}ms{}", model, duration_ms, tokens));
            }
            Event::ToolCall { tool_name, duration_ms, success, error, result_summary, .. } => {
                let status = if *success {
                    "ok".to_string()
                } else {
                    format!("error: {}", error.as_deref().unwrap_or("unknown"))
                };
                let summary = result_summary.as_deref().map(|s| format!(" {}", s)).unwrap_or_default();
                self.logger.tool(&format!("{} {}{} ({}ms)", tool_name, status, summary, duration_ms));
            }
            Event::Retry { operation, attempt, delay_ms } => {
                self.logger.log(&format!("[retry] {} attempt {} after {}ms", operation, attempt, delay_ms));
            }
            Event::Error { context, message } => {
                self.logger.log(&format!("[error] {}: {}", context, message));
            }
        }
    }
}

/// Observer that fans out events to multiple observers.
pub struct MultiObserver {
    observers: Vec<std::sync::Arc<dyn Observer>>,
}

impl MultiObserver {
    /// Create a new multi-observer with the given observers.
    pub fn new(observers: Vec<std::sync::Arc<dyn Observer>>) -> Self {
        Self { observers }
    }
}

#[async_trait]
impl Observer for MultiObserver {
    async fn observe(&self, event: Event) {
        for observer in &self.observers {
            observer.observe(event.clone()).await;
        }
    }
}

/// Truncate a string for display, adding "..." if truncated.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_event_serialization() {
        let event = Event::AgentStart {
            agent_id: "test".to_string(),
            task: "do something".to_string(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("AgentStart"));
        assert!(json.contains("test"));
    }

    #[test]
    fn test_event_serialization_all_variants() {
        let events = vec![
            Event::AgentStart {
                agent_id: "a1".to_string(),
                task: "task".to_string(),
            },
            Event::AgentComplete {
                agent_id: "a1".to_string(),
                duration_ms: 100,
                success: true,
            },
            Event::LlmCall {
                model: "gpt-4".to_string(),
                tokens_in: Some(100),
                tokens_out: Some(50),
                duration_ms: 500,
            },
            Event::ToolCall {
                tool_name: "echo".to_string(),
                duration_ms: 10,
                success: true,
                error: None,
                params: None,
                result_summary: None,
            },
            Event::Retry {
                operation: "llm_call".to_string(),
                attempt: 2,
                delay_ms: 200,
            },
            Event::Error {
                context: "tool".to_string(),
                message: "failed".to_string(),
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            assert!(!json.is_empty());
        }
    }

    #[test]
    fn test_metrics_collector_new() {
        let collector = MetricsCollector::new();
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.llm_calls, 0);
        assert_eq!(snapshot.tool_calls, 0);
        assert_eq!(snapshot.errors, 0);
        assert_eq!(snapshot.total_tokens, 0);
    }

    #[tokio::test]
    async fn test_metrics_collector_counts() {
        let collector = MetricsCollector::new();

        collector.observe(Event::LlmCall {
            model: "test".to_string(),
            tokens_in: Some(100),
            tokens_out: Some(50),
            duration_ms: 500,
        }).await;

        collector.observe(Event::ToolCall {
            tool_name: "echo".to_string(),
            duration_ms: 10,
            success: true,
            error: None,
            params: None,
            result_summary: None,
        }).await;

        collector.observe(Event::ToolCall {
            tool_name: "add".to_string(),
            duration_ms: 5,
            success: false,
            error: Some("failed".to_string()),
            params: None,
            result_summary: None,
        }).await;

        collector.observe(Event::Retry {
            operation: "llm".to_string(),
            attempt: 1,
            delay_ms: 100,
        }).await;

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.llm_calls, 1);
        assert_eq!(snapshot.tool_calls, 2);
        assert_eq!(snapshot.errors, 1); // One failed tool call
        assert_eq!(snapshot.total_tokens, 150); // 100 + 50
        assert_eq!(snapshot.retries, 1);
        assert_eq!(snapshot.llm_time_ms, 500);
        assert_eq!(snapshot.tool_time_ms, 15); // 10 + 5
    }

    #[tokio::test]
    async fn test_console_observer_verbose() {
        // Just ensure it doesn't panic
        let observer = ConsoleObserver::new(true);
        observer.observe(Event::AgentStart {
            agent_id: "test".to_string(),
            task: "do something".to_string(),
        }).await;
        observer.observe(Event::AgentComplete {
            agent_id: "test".to_string(),
            duration_ms: 100,
            success: true,
        }).await;
    }

    #[tokio::test]
    async fn test_console_observer_non_verbose() {
        let observer = ConsoleObserver::new(false);
        // In non-verbose mode, only errors and completions print
        observer.observe(Event::AgentStart {
            agent_id: "test".to_string(),
            task: "do something".to_string(),
        }).await;
        observer.observe(Event::Error {
            context: "test".to_string(),
            message: "something went wrong".to_string(),
        }).await;
    }

    #[tokio::test]
    async fn test_multi_observer() {
        let metrics1 = Arc::new(MetricsCollector::new());
        let metrics2 = Arc::new(MetricsCollector::new());
        let multi = MultiObserver::new(vec![metrics1.clone(), metrics2.clone()]);

        multi.observe(Event::LlmCall {
            model: "test".to_string(),
            tokens_in: Some(10),
            tokens_out: Some(20),
            duration_ms: 100,
        }).await;

        // Both collectors should have recorded the event
        assert_eq!(metrics1.snapshot().llm_calls, 1);
        assert_eq!(metrics2.snapshot().llm_calls, 1);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("hi", 2), "hi");
    }

    #[test]
    fn test_metrics_snapshot_serialization() {
        let snapshot = MetricsSnapshot {
            llm_calls: 5,
            tool_calls: 10,
            errors: 1,
            total_tokens: 1000,
            retries: 2,
            llm_time_ms: 5000,
            tool_time_ms: 100,
        };
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("llm_calls"));
        assert!(json.contains("1000"));
    }
}
