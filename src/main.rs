use anima::tools::{AddTool, EchoTool};
use anima::{Runtime};

#[tokio::main]
async fn main() {
    println!("Creating Runtime...");
    let mut runtime = Runtime::new();

    println!("Spawning agent with id 'demo-agent'...");
    let agent = runtime.spawn_agent("demo-agent".to_string());

    println!("Registering EchoTool and AddTool on the agent...");
    agent.register_tool(Box::new(EchoTool {}));
    agent.register_tool(Box::new(AddTool {}));

    println!("Calling echo tool with input: {{\"message\": \"Hello, Anima!\"}}");
    let echo_result = agent.call_tool("echo", r#"{"message": "Hello, Anima!"}"#).await;
    println!("Echo result: {:?}", echo_result);

    println!("Calling add tool with input: {{\"a\": 5, \"b\": 3}}");
    let add_result = agent.call_tool("add", r#"{"a": 5, "b": 3}"#).await;
    println!("Add result: {:?}", add_result);

    println!("Listing all tools on the agent:");
    // Note: Agent doesn't have list_tools method, so we're skipping this for now
    // let tools = agent.list_tools();
    // for tool in tools {
    //     println!("Tool name: {}", tool.name);
    // }
}
