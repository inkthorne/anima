use async_trait::async_trait;
use crate::error::ToolError;
use crate::tool::Tool;
use serde_json::Value;

/// Tool for making HTTP requests.
#[derive(Debug)]
pub struct HttpTool {
    client: reqwest::Client,
}

impl HttpTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for HttpTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for HttpTool {
    fn name(&self) -> &str {
        "http"
    }

    fn description(&self) -> &str {
        "Makes an HTTP request to the given URL"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to send the request to"
                },
                "method": {
                    "type": "string",
                    "description": "The HTTP method (GET, POST, PUT, DELETE, etc.). Defaults to GET",
                    "default": "GET"
                },
                "body": {
                    "type": "string",
                    "description": "The request body (for POST, PUT, etc.)"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("Missing or invalid 'url' field".to_string()))?;

        let method = input
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET")
            .to_uppercase();

        let body = input
            .get("body")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let request = match method.as_str() {
            "GET" => self.client.get(url),
            "POST" => {
                let req = self.client.post(url);
                if let Some(b) = &body {
                    req.body(b.clone())
                } else {
                    req
                }
            }
            "PUT" => {
                let req = self.client.put(url);
                if let Some(b) = &body {
                    req.body(b.clone())
                } else {
                    req
                }
            }
            "DELETE" => self.client.delete(url),
            "PATCH" => {
                let req = self.client.patch(url);
                if let Some(b) = &body {
                    req.body(b.clone())
                } else {
                    req
                }
            }
            "HEAD" => self.client.head(url),
            _ => return Err(ToolError::InvalidInput(format!("Unsupported HTTP method: {}", method))),
        };

        let response = request
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("HTTP request failed: {}", e)))?;

        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Failed to read response body: {}", e)))?;

        Ok(serde_json::json!({
            "status": status,
            "body": body
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_http_tool_name() {
        let tool = HttpTool::new();
        assert_eq!(tool.name(), "http");
    }

    #[test]
    fn test_http_tool_description() {
        let tool = HttpTool::new();
        assert!(tool.description().contains("HTTP"));
    }

    #[test]
    fn test_http_tool_schema() {
        let tool = HttpTool::new();
        let schema = tool.schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["url"].is_object());
        assert!(schema["properties"]["method"].is_object());
        assert!(schema["properties"]["body"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&json!("url")));
    }

    #[test]
    fn test_http_tool_default() {
        let tool = HttpTool::default();
        assert_eq!(tool.name(), "http");
    }

    #[tokio::test]
    async fn test_http_missing_url() {
        let tool = HttpTool::new();
        let result = tool.execute(json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn test_http_invalid_url() {
        let tool = HttpTool::new();
        let result = tool.execute(json!({"url": "not-a-valid-url"})).await;
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[tokio::test]
    async fn test_http_unsupported_method() {
        let tool = HttpTool::new();
        let result = tool.execute(json!({"url": "https://example.com", "method": "CONNECT"})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    // Network-dependent tests are marked #[ignore]
    #[tokio::test]
    #[ignore]
    async fn test_http_get_request() {
        let tool = HttpTool::new();
        let result = tool.execute(json!({"url": "https://httpbin.org/get"})).await.unwrap();
        assert_eq!(result["status"], 200);
    }

    #[tokio::test]
    #[ignore]
    async fn test_http_post_request() {
        let tool = HttpTool::new();
        let result = tool.execute(json!({
            "url": "https://httpbin.org/post",
            "method": "POST",
            "body": "test data"
        })).await.unwrap();
        assert_eq!(result["status"], 200);
    }
}
