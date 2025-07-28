import sys

sys.path.append("../../")

from typing import Dict, Any, List
from agents.base_agent import (
    BaseAgent,
    AgentRequest,
    AgentResponse,
    AgentCapability,
    TaskType,
)
from tools import discover_tools, execute_tool
from utils import generate_completion_with_tools
from atla_insights import instrument


class SearchAgent(BaseAgent):
    """Agent specialized in searching and retrieving information"""

    def __init__(self):
        super().__init__("SearchAgent")
        self.max_turns = 5

    def get_capabilities(self) -> AgentCapability:
        """Return this agent's capabilities"""
        return AgentCapability(
            name="SearchAgent",
            description="Specialized agent for finding and retrieving information from the web",
            best_for=[
                "Factual questions requiring current information",
                "Finding specific data from websites",
                "Research on recent events or developments",
                "Looking up official information from authoritative sources",
                "Gathering comprehensive information on a topic",
            ],
            example_tasks=[
                "What is the current population of Tokyo?",
                "Find the latest news about renewable energy developments",
                "What are the requirements for a US passport?",
                "Look up the official exchange rate for USD to EUR today",
                "Research the specifications of the latest iPhone model",
            ],
        )

    def can_handle(self, request: AgentRequest) -> bool:
        """Check if this is a search task"""
        return request.task_type == TaskType.SEARCH

    def get_system_prompt(self) -> str:
        return """
You are a specialized search agent. Your job is to find specific information using web searches and reading web pages.

Your capabilities:
- Search Google for information
- Open and read web pages
- Extract relevant information from search results

IMPORTANT:
- Be very careful to use trustable information. Always prefer official websites, open websites to understand the context.
- If initial search doesn't find what you need, try different keywords
- Always verify information by reading the actual web pages, not just search snippets
- Focus on finding the specific information requested

When you have found the answer, provide it clearly and cite your sources.

End your response with "SEARCH_COMPLETE:" followed by your findings.
"""

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get search-related tools only"""
        all_tools = discover_tools()
        search_tools = []

        for tool in all_tools:
            tool_name = tool["function"]["name"]
            if tool_name in ["google_search", "open_webpage", "search_and_read"]:
                search_tools.append(tool)

        return search_tools

    @instrument("SearchAgent - Web Research")
    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute search task"""
        try:
            tools = self.get_tools()

            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": request.task},
            ]

            max_turns = 5
            turn_count = 0

            while turn_count < max_turns:
                response = generate_completion_with_tools(
                    messages, tools, model="gpt-4o"
                )
                turn_count += 1

                # Extract content from OpenAI response
                content = (
                    response.choices[0].message.content if response.choices else ""
                )

                if content and "SEARCH_COMPLETE:" in content:
                    result = content.split("SEARCH_COMPLETE:", 1)[1].strip()
                    return self._create_success_response(result)

                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = eval(tool_call.function.arguments)

                        tool_result = execute_tool(function_name, **function_args)

                        messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": [tool_call],
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(tool_result),
                            }
                        )
                else:
                    messages.append({"role": "assistant", "content": content})

            return AgentResponse(
                result="Search timeout - could not complete search within turn limit",
                success=False,
                error="Max turns exceeded",
            )

        except Exception as e:
            return AgentResponse(result="", success=False, error=str(e))
