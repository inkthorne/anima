//! Embedding client for semantic memory.
//!
//! This module provides an embedding client that uses Ollama to generate
//! text embeddings for semantic memory search.

use std::fmt;

/// Error type for embedding operations.
#[derive(Debug, Clone)]
pub enum EmbeddingError {
    /// HTTP request failed
    RequestError(String),
    /// Failed to parse response
    ParseError(String),
    /// Provider error
    ProviderError(String),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingError::RequestError(msg) => write!(f, "Request error: {}", msg),
            EmbeddingError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            EmbeddingError::ProviderError(msg) => write!(f, "Provider error: {}", msg),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// Client for generating text embeddings via Ollama.
#[derive(Debug, Clone)]
pub struct EmbeddingClient {
    /// Base URL for the Ollama API
    url: String,
    /// Model name for embeddings
    model: String,
}

impl EmbeddingClient {
    /// Create a new embedding client.
    ///
    /// # Arguments
    /// * `model` - The embedding model to use (e.g., "nomic-embed-text")
    /// * `url` - Optional base URL for Ollama (defaults to "http://localhost:11434")
    pub fn new(model: &str, url: Option<&str>) -> Self {
        Self {
            url: url.unwrap_or("http://localhost:11434").to_string(),
            model: model.to_string(),
        }
    }

    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Generate an embedding for a single text.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let client = reqwest::Client::new();
        let url = format!("{}/api/embeddings", self.url);

        let request_body = serde_json::json!({
            "model": self.model,
            "prompt": text
        });

        let response = client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| EmbeddingError::RequestError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::ProviderError(format!(
                "Ollama returned {}: {}",
                status, body
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| EmbeddingError::ParseError(e.to_string()))?;

        let embedding = json
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| EmbeddingError::ParseError("Missing 'embedding' field".to_string()))?;

        let floats: Result<Vec<f32>, _> = embedding
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| EmbeddingError::ParseError("Invalid float in embedding".to_string()))
            })
            .collect();

        floats
    }

    /// Generate embeddings for multiple texts.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }
}

/// Calculate the cosine similarity between two vectors.
///
/// Returns a value between -1.0 and 1.0, where 1.0 means identical,
/// 0.0 means orthogonal, and -1.0 means opposite.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_similar() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.9); // Should be very similar
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_embedding_client_new() {
        let client = EmbeddingClient::new("nomic-embed-text", None);
        assert_eq!(client.url, "http://localhost:11434");
        assert_eq!(client.model, "nomic-embed-text");
    }

    #[test]
    fn test_embedding_client_custom_url() {
        let client = EmbeddingClient::new("nomic-embed-text", Some("http://custom:8080"));
        assert_eq!(client.url, "http://custom:8080");
        assert_eq!(client.model, "nomic-embed-text");
    }

    #[test]
    fn test_embedding_error_display() {
        let err = EmbeddingError::RequestError("connection failed".to_string());
        assert!(err.to_string().contains("connection failed"));

        let err = EmbeddingError::ParseError("invalid json".to_string());
        assert!(err.to_string().contains("invalid json"));

        let err = EmbeddingError::ProviderError("model not found".to_string());
        assert!(err.to_string().contains("model not found"));
    }
}
