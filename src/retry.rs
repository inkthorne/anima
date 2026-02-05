//! Retry policy infrastructure for resilient error recovery.
//!
//! Provides exponential backoff with jitter for transient failures.

use rand::Rng;
use std::future::Future;
use std::time::Duration;

/// Configuration for retry behavior with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries, just the initial attempt)
    pub max_retries: usize,
    /// Initial delay before first retry in milliseconds
    pub initial_delay_ms: u64,
    /// Maximum delay cap in milliseconds
    pub max_delay_ms: u64,
    /// Base for exponential backoff (typically 2.0)
    pub exponential_base: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            exponential_base: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a policy that never retries
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set initial delay in milliseconds
    pub fn with_initial_delay_ms(mut self, delay_ms: u64) -> Self {
        self.initial_delay_ms = delay_ms;
        self
    }

    /// Set maximum delay cap in milliseconds
    pub fn with_max_delay_ms(mut self, max_delay_ms: u64) -> Self {
        self.max_delay_ms = max_delay_ms;
        self
    }

    /// Set exponential base (typically 2.0)
    pub fn with_exponential_base(mut self, base: f64) -> Self {
        self.exponential_base = base;
        self
    }

    /// Calculate delay for a given attempt (0-indexed).
    /// Uses exponential backoff with jitter to prevent thundering herd.
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        // Calculate base delay: initial_delay * base^(attempt-1)
        let base_delay =
            self.initial_delay_ms as f64 * self.exponential_base.powi((attempt - 1) as i32);

        // Cap at max delay
        let capped_delay = base_delay.min(self.max_delay_ms as f64);

        // Add jitter: random value between 0 and 50% of the delay
        let mut rng = rand::rng();
        let jitter = rng.random_range(0.0..0.5) * capped_delay;
        let final_delay = capped_delay + jitter;

        Duration::from_millis(final_delay as u64)
    }
}

/// Result of a retry operation, containing context about attempts made.
#[derive(Debug, Clone)]
pub struct RetryResult<T, E> {
    /// The final result (success or last error)
    pub result: Result<T, E>,
    /// Number of attempts made (1 = succeeded on first try)
    pub attempts: usize,
}

/// Execute an async operation with retry logic.
///
/// The operation will be retried according to the policy when it fails
/// with a retryable error. The `is_retryable` function determines which
/// errors should trigger a retry.
///
/// # Example
/// ```ignore
/// let policy = RetryPolicy::default();
/// let result = with_retry(&policy, || async { make_api_call().await }, |e| e.is_transient()).await;
/// ```
pub async fn with_retry<F, Fut, T, E>(
    policy: &RetryPolicy,
    mut operation: F,
    is_retryable: impl Fn(&E) -> bool,
) -> RetryResult<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempts = 0;
    let max_attempts = policy.max_retries + 1; // +1 for initial attempt

    loop {
        attempts += 1;
        let result = operation().await;

        match result {
            Ok(value) => {
                return RetryResult {
                    result: Ok(value),
                    attempts,
                };
            }
            Err(error) => {
                // Check if we should retry
                if attempts < max_attempts && is_retryable(&error) {
                    let delay = policy.delay_for_attempt(attempts);
                    tokio::time::sleep(delay).await;
                    continue;
                }

                // No more retries, return the error
                return RetryResult {
                    result: Err(error),
                    attempts,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_default_policy() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.initial_delay_ms, 100);
        assert_eq!(policy.max_delay_ms, 5000);
        assert!((policy.exponential_base - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_no_retry_policy() {
        let policy = RetryPolicy::no_retry();
        assert_eq!(policy.max_retries, 0);
    }

    #[test]
    fn test_builder_pattern() {
        let policy = RetryPolicy::new()
            .with_max_retries(5)
            .with_initial_delay_ms(200)
            .with_max_delay_ms(10000)
            .with_exponential_base(3.0);

        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.initial_delay_ms, 200);
        assert_eq!(policy.max_delay_ms, 10000);
        assert!((policy.exponential_base - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_delay_for_attempt_zero() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.delay_for_attempt(0), Duration::ZERO);
    }

    #[test]
    fn test_delay_exponential_growth() {
        let policy = RetryPolicy::new()
            .with_initial_delay_ms(100)
            .with_max_delay_ms(10000)
            .with_exponential_base(2.0);

        // Without jitter, delays would be: 100, 200, 400, 800...
        // With jitter (0-50%), delays are in ranges: 100-150, 200-300, 400-600...

        let delay1 = policy.delay_for_attempt(1);
        let delay2 = policy.delay_for_attempt(2);
        let delay3 = policy.delay_for_attempt(3);

        // Verify rough exponential growth (accounting for jitter)
        assert!(delay1.as_millis() >= 100 && delay1.as_millis() <= 150);
        assert!(delay2.as_millis() >= 200 && delay2.as_millis() <= 300);
        assert!(delay3.as_millis() >= 400 && delay3.as_millis() <= 600);
    }

    #[test]
    fn test_delay_respects_max() {
        let policy = RetryPolicy::new()
            .with_initial_delay_ms(1000)
            .with_max_delay_ms(2000)
            .with_exponential_base(10.0);

        // Even with base=10, delay should be capped at max_delay + jitter
        let delay = policy.delay_for_attempt(5);
        // Max is 2000, with up to 50% jitter = 3000 max
        assert!(delay.as_millis() <= 3000);
    }

    #[tokio::test]
    async fn test_with_retry_succeeds_first_try() {
        let policy = RetryPolicy::default();
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = call_count.clone();

        let result = with_retry(
            &policy,
            || {
                let c = count.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, String>("success")
                }
            },
            |_: &String| true,
        )
        .await;

        assert!(result.result.is_ok());
        assert_eq!(result.attempts, 1);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_succeeds_after_retries() {
        let policy = RetryPolicy::new()
            .with_max_retries(3)
            .with_initial_delay_ms(1); // Fast for testing
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = call_count.clone();

        let result = with_retry(
            &policy,
            || {
                let c = count.clone();
                async move {
                    let attempt = c.fetch_add(1, Ordering::SeqCst) + 1;
                    if attempt < 3 {
                        Err("transient error".to_string())
                    } else {
                        Ok("success")
                    }
                }
            },
            |_: &String| true, // All errors are retryable
        )
        .await;

        assert!(result.result.is_ok());
        assert_eq!(result.attempts, 3);
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_exhausts_retries() {
        let policy = RetryPolicy::new()
            .with_max_retries(2)
            .with_initial_delay_ms(1);
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = call_count.clone();

        let result = with_retry(
            &policy,
            || {
                let c = count.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>("persistent error".to_string())
                }
            },
            |_: &String| true,
        )
        .await;

        assert!(result.result.is_err());
        assert_eq!(result.attempts, 3); // 1 initial + 2 retries
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_non_retryable_fails_fast() {
        let policy = RetryPolicy::new()
            .with_max_retries(5)
            .with_initial_delay_ms(1);
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = call_count.clone();

        let result = with_retry(
            &policy,
            || {
                let c = count.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>("non-retryable error")
                }
            },
            |_: &&str| false, // No errors are retryable
        )
        .await;

        assert!(result.result.is_err());
        assert_eq!(result.attempts, 1); // No retries
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_no_retry_policy() {
        let policy = RetryPolicy::no_retry();
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = call_count.clone();

        let result = with_retry(
            &policy,
            || {
                let c = count.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>("error")
                }
            },
            |_: &&str| true,
        )
        .await;

        assert!(result.result.is_err());
        assert_eq!(result.attempts, 1);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }
}
