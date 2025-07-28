import sys
import json
from typing import Dict, Any, List
from agents.base_agent import (
    BaseAgent,
    AgentRequest,
    AgentResponse,
    TaskType,
    AgentCapability,
)
from tool_discovery import discover_tools, execute_tool
from utils import generate_completion_with_tools
from atla_insights import instrument


class ReasoningAgent(BaseAgent):
    """Agent specialized in logical reasoning and problem solving"""

    def __init__(self):
        super().__init__("ReasoningAgent")
        self.max_turns = 5

    def get_capabilities(self) -> AgentCapability:
        """Return this agent's capabilities"""
        return AgentCapability(
            name="ReasoningAgent",
            description="Specialized agent for logical analysis, mathematical calculations, and systematic problem solving",
            best_for=[
                "Mathematical calculations and problem solving",
                "Logical reasoning and deduction",
                "Step-by-step analysis of complex problems",
                "Critical thinking and argument evaluation",
                "Puzzles and brain teasers requiring systematic approach",
            ],
            example_tasks=[
                "Solve this math problem: If a train travels 120 km in 2 hours, what is its speed?",
                "Analyze the logic in this argument: All cats are animals. Fluffy is a cat. Therefore...",
                "A standard Rubik's cube puzzle with missing pieces - determine what's missing",
                "Calculate compound interest on $1000 at 5% for 3 years",
                "Evaluate the reasoning: If all birds can fly, and penguins are birds, can penguins fly?",
            ],
        )

    def get_system_prompt(self) -> str:
        return """You are a specialized reasoning agent focused on logical analysis and problem solving.

Your process:
1. Break down the problem systematically
2. Apply logical reasoning step by step
3. Use calculations when needed
4. MANDATORY: State "Let me double-check this reasoning:" and critically examine your logic
5. Consider alternative approaches and potential flaws
6. Provide your final conclusion

Focus on:
- Clear logical steps
- Identifying assumptions
- Mathematical accuracy
- Critical self-evaluation

When complete, end with "REASONING_COMPLETE:" followed by your final answer."""

    def can_handle(self, request: AgentRequest) -> bool:
        """Check if this is a reasoning task"""
        return request.task_type == TaskType.REASONING

    def _get_reasoning_tools(self) -> List[Dict[str, Any]]:
        """Get reasoning-related tools only"""
        all_tools = discover_tools()
        reasoning_tool_names = {"calculate"}

        return [
            tool
            for tool in all_tools
            if tool["function"]["name"] in reasoning_tool_names
        ]

    @instrument("ReasoningAgent - Logic & Calculations")
    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute reasoning task"""
        try:
            tools = self._get_reasoning_tools()
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": request.task},
            ]

            for turn in range(self.max_turns):
                response = generate_completion_with_tools(
                    messages, tools, model="gpt-4o"
                )

                # Extract content from OpenAI response
                content = (
                    response.choices[0].message.content if response.choices else ""
                )

                # Show reasoning process
                if content:
                    print(f"\n🧠 REASONING: {content}")

                # Check for completion
                if content and "REASONING_COMPLETE:" in content:
                    result = content.split("REASONING_COMPLETE:", 1)[1].strip()
                    return self._create_success_response(
                        result, {"turns_used": turn + 1, "reasoning_complete": True}
                    )

                # Execute tool calls
                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        tool_result = execute_tool(function_name, **function_args)

                        messages.extend(
                            [
                                {
                                    "role": "assistant",
                                    "content": content,
                                    "tool_calls": [tool_call],
                                },
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(tool_result),
                                },
                            ]
                        )
                else:
                    messages.append({"role": "assistant", "content": content})

            return self._create_error_response(
                "Reasoning timeout - could not complete within turn limit",
                {"turns_used": self.max_turns},
            )

        except Exception as e:
            return self._create_error_response(f"Reasoning execution failed: {str(e)}")
