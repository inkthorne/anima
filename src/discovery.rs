//! Daemon discovery for finding running agent daemons.
//!
//! This module provides functions to discover and connect to running agent daemons
//! by scanning PID files in the ~/.anima/agents/ directory.

use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct DaemonInfo {
    pub name: String,
    pub pid: u32,
    pub socket_path: String,
    pub is_alive: bool,
}

/// Info about a running agent daemon (for REPL/CLI use)
#[derive(Debug, Clone)]
pub struct RunningAgent {
    pub name: String,
    pub socket_path: PathBuf,
}

/// Discover all running agent daemons.
///
/// Scans ~/.anima/agents/*/daemon.pid files and checks if each process is alive.
/// Returns a list of running daemons with their socket paths.
pub fn discover_daemons() -> Vec<DaemonInfo> {
    let mut daemons = Vec::new();
    let agents_dir = dirs::home_dir()
        .map(|h| h.join(".anima").join("agents"))
        .unwrap_or_else(|| PathBuf::from("~/.anima/agents"));

    if !agents_dir.exists() {
        return daemons;
    }

    // Iterate over all directories in the agents directory
    if let Ok(entries) = fs::read_dir(&agents_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let agent_dir = entry.path();

                // Skip if not a directory
                if !agent_dir.is_dir() {
                    continue;
                }

                // Get agent name from directory name
                let name = agent_dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                // Check for daemon.pid file
                let pid_file = agent_dir.join("daemon.pid");
                if pid_file.exists() {
                    if let Ok(pid_content) = fs::read_to_string(&pid_file) {
                        if let Ok(pid) = pid_content.trim().parse::<u32>() {
                            // Check if process is alive
                            let is_alive = is_process_alive(pid);

                            // Construct socket path
                            let socket_path =
                                agent_dir.join("agent.sock").to_string_lossy().to_string();

                            daemons.push(DaemonInfo {
                                name,
                                pid,
                                socket_path,
                                is_alive,
                            });
                        }
                    }
                }
            }
        }
    }

    daemons
}

/// Get the agents directory path (~/.anima/agents/)
fn agents_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".anima").join("agents"))
        .unwrap_or_else(|| PathBuf::from("~/.anima/agents"))
}

/// Discover all running agent daemons, returning RunningAgent structs.
pub fn discover_running_agents() -> Vec<RunningAgent> {
    discover_daemons()
        .into_iter()
        .filter(|d| d.is_alive)
        .map(|d| RunningAgent {
            name: d.name,
            socket_path: PathBuf::from(d.socket_path),
        })
        .collect()
}

/// Check if an agent is currently running.
pub fn is_agent_running(name: &str) -> bool {
    let pid_file = agents_dir().join(name).join("daemon.pid");
    if !pid_file.exists() {
        return false;
    }
    if let Ok(content) = fs::read_to_string(&pid_file) {
        if let Ok(pid) = content.trim().parse::<u32>() {
            return is_process_alive(pid);
        }
    }
    false
}

/// Get info about a running agent by name.
pub fn get_running_agent(name: &str) -> Option<RunningAgent> {
    if is_agent_running(name) {
        Some(RunningAgent {
            name: name.to_string(),
            socket_path: agents_dir().join(name).join("agent.sock"),
        })
    } else {
        None
    }
}

/// Get the socket path for an agent.
pub fn agent_socket_path(name: &str) -> Option<PathBuf> {
    let path = agents_dir().join(name).join("agent.sock");
    Some(path)
}

/// List all saved agents (directories with config.toml in ~/.anima/agents/).
pub fn list_saved_agents() -> Vec<String> {
    let agents_path = agents_dir();
    if !agents_path.exists() {
        return Vec::new();
    }

    match fs::read_dir(&agents_path) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter(|e| e.path().join("config.toml").exists())
            .filter_map(|e| e.file_name().to_str().map(String::from))
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Check if an agent exists (has a config.toml in ~/.anima/agents/{name}/).
pub fn agent_exists(name: &str) -> bool {
    let config_path = agents_dir().join(name).join("config.toml");
    config_path.exists()
}

#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    // Use kill(pid, 0) to check if process exists
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(windows)]
fn is_process_alive(pid: u32) -> bool {
    // On Windows, we check if the process exists using Windows API
    use winapi::um::errhandlingapi::GetLastError;
    use winapi::um::processthreadsapi::OpenProcess;
    use winapi::um::processthreadsapi::PROCESS_QUERY_INFORMATION;
    use winapi::um::winnt::INVALID_HANDLE_VALUE;
    use winapi::um::winnt::SYNCHRONIZE;

    let handle = unsafe { OpenProcess(PROCESS_QUERY_INFORMATION | SYNCHRONIZE, 0, pid) };

    if handle.is_null() {
        false
    } else {
        unsafe {
            winapi::um::processthreadsapi::CloseHandle(handle);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_daemons_returns_vec() {
        // Test that discover_daemons returns a Vec (may or may not be empty
        // depending on whether daemons are running on the system)
        let daemons = discover_daemons();
        // Just verify it returns without panicking
        let _ = daemons;
    }

    #[test]
    fn test_daemon_info_struct() {
        let daemon = DaemonInfo {
            name: "test".to_string(),
            pid: 12345,
            socket_path: "/tmp/test.sock".to_string(),
            is_alive: true,
        };
        assert_eq!(daemon.name, "test");
        assert_eq!(daemon.pid, 12345);
        assert_eq!(daemon.is_alive, true);
    }
}
