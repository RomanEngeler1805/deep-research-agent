# Multi-Agent AI Research Assistant

This project implements a multi-agent AI system designed to perform deep research by combining web search and reasoning capabilities. An orchestrator agent delegates tasks to specialized sub-agents to provide comprehensive and well-reasoned answers to user queries.

## Features

- **Multi-Agent Architecture**: An orchestrator agent plans and delegates tasks to a `SearchAgent` (for web searches) and a `ReasoningAgent` (for logical tasks).
- **Dynamic Tool Discovery**: Tools available in `actions.py` are automatically discovered and made available to the agents.
- **CLI Interface**: Interact with the agent directly from your command line.
- **Observability**: Integrated with Atla Insights for tracing and monitoring agent behavior.
- **Batch Processing**: Run the agent on a list of questions from a file.

## Setup and Installation

This project uses `uv` for Python package management.

1.  **Install `uv`**:
    If you don't have `uv` installed, follow the official installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/)

2.  **Create a virtual environment and install dependencies**:
    ```bash
    uv sync
    ```

3.  **Set up environment variables**:
    Create a `.env` file in the root of the project and add your API keys:
    ```
    OPENAI_API_KEY="..."
    GOOGLE_API_KEY="..."
    GOOGLE_SEARCH_ENGINE_ID="..."
    ATLA_INSIGHTS_TOKEN="..."
    ```

## Usage

You can run the agent in two modes: interactive or single-query.

### Interactive Mode

To start an interactive session with the agent, run:

```bash
uv run python multi_agent_main.py
```

You can then type your questions directly into the terminal.

### Single Query Mode

To run a single query and get a final answer:

```bash
uv run python multi_agent_main.py "Your question here"
```

## Example Questions

A few example questions from the GAIA Level 1 benchmark can be found in `questions.txt`. You can use these to test the agent's capabilities.