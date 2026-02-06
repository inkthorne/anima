use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const AUTH_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const SCOPE: &str = "org:create_api_key user:profile user:inference";
const EXPIRY_BUFFER_SECS: i64 = 300;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StoredTokens {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: i64,
}

// ---------------------------------------------------------------------------
// Token file helpers
// ---------------------------------------------------------------------------

pub fn auth_file_path() -> PathBuf {
    dirs::home_dir()
        .expect("Could not determine home directory")
        .join(".anima")
        .join("auth.json")
}

pub fn save_tokens(tokens: &StoredTokens) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = auth_file_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(tokens)?;
    std::fs::write(&path, json)?;
    // Restrict permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub fn load_tokens() -> Option<StoredTokens> {
    let path = auth_file_path();
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

pub fn clear_tokens() {
    let path = auth_file_path();
    let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// PKCE helpers
// ---------------------------------------------------------------------------

pub fn generate_code_verifier() -> String {
    use rand::Rng;
    let mut rng = rand::rng();
    let bytes: Vec<u8> = (0..96).map(|_| rng.random::<u8>()).collect();
    URL_SAFE_NO_PAD.encode(&bytes)
}

pub fn generate_code_challenge(verifier: &str) -> String {
    let hash = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(hash)
}

// ---------------------------------------------------------------------------
// OAuth flow
// ---------------------------------------------------------------------------

pub fn build_auth_url() -> (String, String, String) {
    let verifier = generate_code_verifier();
    let challenge = generate_code_challenge(&verifier);

    use rand::Rng;
    let state_bytes: Vec<u8> = (0..32).map(|_| rand::rng().random::<u8>()).collect();
    let state = URL_SAFE_NO_PAD.encode(&state_bytes);

    let url = format!(
        "{}?response_type=code&client_id={}&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method=S256&state={}",
        AUTH_URL,
        urlencoding(CLIENT_ID),
        urlencoding(REDIRECT_URI),
        urlencoding(SCOPE),
        urlencoding(&challenge),
        urlencoding(&state),
    );

    (url, verifier, state)
}

/// Extract a human-readable error from a token endpoint response body.
/// Handles both standard OAuth (`error_description`/`error`) and
/// Anthropic's nested format (`error.message`/`error.type`).
fn extract_error_message(text: &str) -> String {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|b| {
            b["error_description"]
                .as_str()
                .or(b["error"]["message"].as_str())
                .or(b["error"].as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| text.to_string())
}

/// Minimal percent-encoding for URL query parameters in `build_auth_url()`.
/// POST body encoding uses `reqwest::RequestBuilder::json()` instead.
fn urlencoding(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('&', "%26")
        .replace('=', "%3D")
        .replace('+', "%2B")
        .replace('#', "%23")
        .replace('?', "%3F")
}

pub async fn exchange_code(
    code: &str,
    state: &str,
    expected_state: &str,
    verifier: &str,
) -> Result<StoredTokens, Box<dyn std::error::Error + Send + Sync>> {
    if state != expected_state {
        return Err("OAuth state mismatch — possible CSRF attack".into());
    }

    let client = Client::new();
    let body = serde_json::json!({
        "grant_type": "authorization_code",
        "code": code,
        "state": state,
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    });
    let response = client.post(TOKEN_URL).json(&body).send().await?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let err = extract_error_message(&text);
        return Err(format!("Token exchange failed ({}): {}", status, err).into());
    }
    let body: serde_json::Value = response.json().await?;

    let expires_in = body["expires_in"].as_i64().unwrap_or(3600);
    let now = chrono::Utc::now().timestamp();

    Ok(StoredTokens {
        access_token: body["access_token"]
            .as_str()
            .ok_or("Missing access_token in response")?
            .to_string(),
        refresh_token: body["refresh_token"]
            .as_str()
            .ok_or("Missing refresh_token in response")?
            .to_string(),
        expires_at: now + expires_in,
    })
}

pub async fn refresh_tokens(
    refresh_token: &str,
) -> Result<StoredTokens, Box<dyn std::error::Error + Send + Sync>> {
    let client = Client::new();
    let body = serde_json::json!({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
    });
    let response = client.post(TOKEN_URL).json(&body).send().await?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let err = extract_error_message(&text);
        return Err(format!("Token refresh failed ({}): {}", status, err).into());
    }
    let body: serde_json::Value = response.json().await?;

    let expires_in = body["expires_in"].as_i64().unwrap_or(3600);
    let now = chrono::Utc::now().timestamp();

    Ok(StoredTokens {
        access_token: body["access_token"]
            .as_str()
            .ok_or("Missing access_token in refresh response")?
            .to_string(),
        refresh_token: body["refresh_token"]
            .as_str()
            .unwrap_or(refresh_token)
            .to_string(),
        expires_at: now + expires_in,
    })
}

// ---------------------------------------------------------------------------
// Token manager
// ---------------------------------------------------------------------------

pub async fn get_valid_token() -> Result<Option<String>, Box<dyn std::error::Error + Send + Sync>> {
    let Some(tokens) = load_tokens() else {
        return Ok(None);
    };

    let now = chrono::Utc::now().timestamp();
    if now < tokens.expires_at - EXPIRY_BUFFER_SECS {
        return Ok(Some(tokens.access_token));
    }

    // Token expired or about to expire — refresh
    match refresh_tokens(&tokens.refresh_token).await {
        Ok(new_tokens) => {
            save_tokens(&new_tokens)?;
            Ok(Some(new_tokens.access_token))
        }
        Err(e) => {
            // Refresh failed — clear stale tokens
            clear_tokens();
            Err(format!("Subscription token expired and refresh failed: {}. Run `anima login` again.", e).into())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_verifier_length() {
        let v = generate_code_verifier();
        assert!(v.len() >= 43, "verifier too short: {}", v.len());
        assert!(v.len() <= 128, "verifier too long: {}", v.len());
    }

    #[test]
    fn test_code_challenge_is_base64url() {
        let v = generate_code_verifier();
        let c = generate_code_challenge(&v);
        // SHA-256 = 32 bytes → 43 base64url chars (no padding)
        assert_eq!(c.len(), 43);
        assert!(!c.contains('+'));
        assert!(!c.contains('/'));
        assert!(!c.contains('='));
    }

    #[test]
    fn test_build_auth_url_contains_required_params() {
        let (url, _verifier, state) = build_auth_url();
        assert!(url.starts_with(AUTH_URL));
        assert!(url.contains("response_type=code"));
        assert!(url.contains(&format!("client_id={}", urlencoding(CLIENT_ID))));
        assert!(url.contains("code_challenge="));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains(&format!("state={}", urlencoding(&state))));
    }

    #[test]
    fn test_urlencoding_special_chars() {
        assert_eq!(urlencoding("hello world"), "hello%20world");
        assert_eq!(urlencoding("a&b=c"), "a%26b%3Dc");
    }

    #[test]
    fn test_auth_file_path() {
        let path = auth_file_path();
        assert!(path.ends_with(".anima/auth.json"));
    }

    #[test]
    fn test_stored_tokens_serde() {
        let tokens = StoredTokens {
            access_token: "acc".to_string(),
            refresh_token: "ref".to_string(),
            expires_at: 1234567890,
        };
        let json = serde_json::to_string(&tokens).unwrap();
        let parsed: StoredTokens = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.access_token, "acc");
        assert_eq!(parsed.refresh_token, "ref");
        assert_eq!(parsed.expires_at, 1234567890);
    }
}
