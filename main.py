from openai import OpenAI
import os
import subprocess
from dotenv import load_dotenv
from prompts import system_prompt
from tool_discovery import discover_tools, execute_tool
from atla_insights import (
    configure,
    instrument_openai,
    instrument,
)
import json
import sys

# Load environment variables
load_dotenv()


def get_git_commit():
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_commit_message():
    """Get the current git commit message."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# Configure Atla Insights - REQUIRED FIRST
metadata = {
    "git_commit": get_git_commit(),
    "git_commit_message": get_git_commit_message(),
    "model": "gpt-4o",
    "environment": "development",
    "agent_version": "1.0.0",
}

configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"), metadata=metadata)

# Instrument OpenAI
instrument_openai()

# Create an instance of the OpenAI class
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_completion_with_tools(messages, tools, model="gpt-4o"):
    """Generate completion from OpenAI API with function calling."""
    response = openai_client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto"
    )
    return response.choices[0].message


def print_step_header(step_number: int, title: str):
    """Print a clear step header."""
    print(f"\n{'='*60}")
    print(f"STEP {step_number}: {title}")
    print(f"{'='*60}")


def print_thought(text: str):
    """Print a thought in a clean format."""
    if text and text.strip():
        print(f"\n💭 REASONING:")
        print(f"   {text}")


def print_action(action_name: str, input_text: str):
    """Print an action in a clean format."""
    print(f"\n🔧 TOOL CALL:")
    print(f"   Function: {action_name}")
    print(f"   Arguments: {input_text}")


def print_observation(text: str, max_length: int = 800):
    """Print an observation in a clean format."""
    print(f"\n📊 RESULT:")
    if len(text) > max_length:
        print(f"   {text[:max_length]}...")
        print(f"   [Content truncated - showing first {max_length} characters]")
    else:
        # Split into lines and indent each line
        lines = text.split("\n")
        for line in lines:
            print(f"   {line}")


@instrument("Process query internally")
def process_query_internally(
    query: str, max_turns: int = 10, show_thinking: bool = True
) -> str:
    """
    Process a query internally, allowing multiple tool executions.

    Args:
        query: The user's query
        max_turns: Maximum number of tool execution turns
        show_thinking: Whether to display intermediate thinking

    Returns:
        Final answer to the query
    """
    try:
        # Discover available tools
        tools = discover_tools()

        # Start fresh conversation for each query
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": query},
        ]

        turn_count = 0
        final_answer = None

        while turn_count < max_turns:
            # Generate response with function calling
            response = generate_completion_with_tools(messages, tools, model="gpt-4o")

            # Show the assistant's thinking if there's content
            if response.content and show_thinking:
                print_thought(response.content)

            # Check if the model wants to call a function
            if response.tool_calls:
                # Execute each tool call
                for tool_call in response.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    turn_count += 1
                    print_step_header(turn_count, f"Using {function_name}")

                    if show_thinking:
                        print_action(function_name, str(function_args))

                    # Execute the tool
                    try:
                        result = execute_tool(function_name, **function_args)
                    except Exception as e:
                        result = f"Error executing {function_name}: {str(e)}"

                    if show_thinking:
                        print_observation(result)

                    # Add the tool call and result to messages
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.content,
                            "tool_calls": [tool_call],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )

                continue
            else:
                # No tool calls, check if this is the final answer
                if response.content and "FINAL_ANSWER:" in response.content:
                    # Extract the final answer part
                    final_answer = response.content.split("FINAL_ANSWER:", 1)[1].strip()
                    break
                else:
                    # Continue the conversation - agent is reasoning but not done yet
                    messages.append({"role": "assistant", "content": response.content})
                    continue

        # If we hit max turns without a final answer, get one more response
        if final_answer is None:
            final_response = generate_completion_with_tools(
                messages, tools, model="gpt-4o"
            )
            final_answer = final_response.content

        return final_answer

    except Exception as e:
        raise


@instrument("Run interactive agent mode")
def run_interactive_mode():
    """Run the agent in interactive mode."""
    print("=" * 80)
    print("🤖 AI Agent Ready")
    print("=" * 80)
    print("Ask me anything! I'll research and provide a comprehensive answer.")
    print("Type 'quit' or 'exit' to end the session.\n")

    try:
        # Get user input
        user_input = input("\n❓ You: ").strip()

        print(f"\n🔄 RESEARCHING: {user_input}")
        print("=" * 80)

        # Process the query internally - each query starts fresh
        final_answer = process_query_internally(user_input, show_thinking=True)

        print("\n" + "=" * 80)
        print("💬 FINAL ANSWER")
        print("=" * 80)
        print(final_answer)
        print("=" * 80)

        # Clear any potential leftover processing
        print()  # Add a blank line for separation

    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please try again.")


@instrument("Run single query")
def run_single_query(query: str):
    """Run a single query without interactive mode."""
    print(f"\n🔄 RESEARCHING: {query}")
    print("=" * 80)

    final_answer = process_query_internally(query, show_thinking=True)

    print("\n" + "=" * 80)
    print("💬 FINAL ANSWER")
    print("=" * 80)
    print(final_answer)
    print("=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_single_query(query)
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()
