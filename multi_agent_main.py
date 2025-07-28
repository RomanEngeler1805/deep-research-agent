import os
import subprocess
import sys
from dotenv import load_dotenv
from atla_insights import (
    configure,
    instrument_openai,
    instrument,
    mark_success,
    mark_failure,
)
from agents import OrchestratorAgent, AgentRequest

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


def setup_observability():
    """Configure Atla Insights with metadata."""
    metadata = {
        "git_commit": get_git_commit(),
        "git_commit_message": get_git_commit_message(),
        "model": "gpt-4o",
        "environment": "development",
        "agent_version": "2.0.0-multi-agent",
        "architecture": "multi-agent-orchestrated",
    }
    configure(token=os.getenv("ATLA_INSIGHTS_TOKEN"), metadata=metadata)
    instrument_openai()


@instrument("Run single query multi-agent")
def run_single_query(query: str):
    """Run a single query through the multi-agent system."""
    print("=" * 80)
    print("🤖 Multi-Agent AI System")
    print("=" * 80)
    print(f"Query: {query}")
    print("-" * 80)

    try:
        orchestrator = OrchestratorAgent()
        request = AgentRequest(task=query)

        print("🧠 Orchestrator analyzing task...")
        result = orchestrator.execute(request)

        print("\n" + "=" * 80)
        print("🎯 FINAL ANSWER:")
        print("=" * 80)
        print(result.result)

        if result.success:
            mark_success()
        else:
            mark_failure()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        mark_failure()


@instrument("Run interactive multi-agent mode")
def run_interactive():
    """Run the multi-agent system in interactive mode."""
    print("=" * 80)
    print("🤖 Multi-Agent AI System Ready")
    print("=" * 80)
    print(
        "Ask me anything! I'll coordinate specialized agents to research and provide comprehensive answers."
    )
    print("Type 'quit' or 'exit' to end the session.\n")

    orchestrator = OrchestratorAgent()

    try:
        # Get user input
        user_input = input("You: ").strip()

        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            return

        if not user_input:
            return

        print("-" * 80)
        print("🧠 Orchestrator analyzing task...")

        request = AgentRequest(task=user_input)
        result = orchestrator.execute(request)

        print("\n" + "=" * 80)
        print("🎯 ANSWER:")
        print("=" * 80)
        print(result.result)
        print()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def main():
    """Main entry point for multi-agent system."""
    setup_observability()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_single_query(query)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
