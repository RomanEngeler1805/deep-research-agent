import json
from typing import Dict, Any, List, Optional
from agents.base_agent import (
    BaseAgent,
    AgentRequest,
    AgentResponse,
    TaskType,
    AgentCapability,
)
from agents.search_agent.agent import SearchAgent
from agents.reasoning_agent.agent import ReasoningAgent
from utils import generate_completion_with_tools
from atla_insights import instrument


class OrchestratorAgent(BaseAgent):
    """Intelligent orchestrator that coordinates specialized sub-agents"""

    def __init__(self):
        super().__init__("OrchestratorAgent")
        self.search_agent = SearchAgent()
        self.reasoning_agent = ReasoningAgent()
        self.agents = {"search": self.search_agent, "reasoning": self.reasoning_agent}
        self.max_turns = 8

    def get_capabilities(self) -> AgentCapability:
        """Return orchestrator capabilities"""
        return AgentCapability(
            name="OrchestratorAgent",
            description="Coordinates and manages specialized agents to solve complex multi-step tasks",
            best_for=[
                "Complex queries requiring multiple types of expertise",
                "Tasks needing both research and analysis",
                "Multi-step problem solving",
                "Coordinating different specialized capabilities",
            ],
            example_tasks=[
                "Research the latest climate data and analyze the trends",
                "Find information about quantum computing and explain how it works",
                "Look up the GDP of Japan and calculate the per capita income",
            ],
        )

    def get_system_prompt(self) -> str:
        """Dynamically generate system prompt from available agents"""
        agent_descriptions = []

        for agent_key, agent in self.agents.items():
            capability = agent.get_capabilities()

            description = f"""**{capability.name}**:
- {capability.description}
- Best for: {', '.join(capability.best_for)}
- Example tasks: {'; '.join(capability.example_tasks[:3])}"""

            agent_descriptions.append(description)

        agents_text = "\n\n".join(agent_descriptions)

        return f"""You are an intelligent orchestrator that coordinates specialized agents to solve complex tasks.

Available agents and their capabilities:

{agents_text}

**Your approach**:
1. Analyze what the user is asking for
2. Determine which specialized agent(s) are best suited for the task
3. Delegate to the appropriate agent(s) based on their capabilities
4. You can use agents sequentially if needed
5. Combine results intelligently

**Delegation format**:
- To use SearchAgent: "DELEGATE_SEARCH: [specific task for search agent]"
- To use ReasoningAgent: "DELEGATE_REASONING: [specific task for reasoning agent]"
- When you have everything needed: "FINAL_ANSWER: [your complete response]"

**Key principles**:
- ALWAYS delegate tasks that match an agent's specialty area
- For mathematical calculations, logic puzzles, step-by-step analysis: use ReasoningAgent
- For factual information, current data, web research: use SearchAgent
- Only handle very simple conversational tasks directly (greetings, clarifications)
- Be specific about what you want each agent to do
- Choose agents based on their described capabilities and strengths
- Combine results from multiple agents when beneficial
"""

    def can_handle(self, request: AgentRequest) -> bool:
        """Orchestrator can handle any request"""
        return True

    def _delegate_to_agent(self, task: str, agent_type: str) -> AgentResponse:
        """Delegate task to specified agent"""
        if agent_type in self.agents:
            if agent_type == "search":
                request = AgentRequest(task=task, task_type=TaskType.SEARCH)
            elif agent_type == "reasoning":
                request = AgentRequest(task=task, task_type=TaskType.REASONING)
            else:
                request = AgentRequest(task=task)

            return self.agents[agent_type].execute(request)
        else:
            return AgentResponse(
                result="Unknown agent type",
                success=False,
                error=f"No agent of type '{agent_type}' available. Available: {list(self.agents.keys())}",
            )

    @instrument("Orchestrator - Task Coordination")
    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute orchestration with intelligent delegation"""
        try:
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": request.task},
            ]

            delegations_made = 0

            for turn in range(self.max_turns):
                response = generate_completion_with_tools(messages, [], model="gpt-4o")

                # Extract content from OpenAI response
                content = (
                    response.choices[0].message.content if response.choices else ""
                )

                print(f"\n🎯 ORCHESTRATOR: {content}")

                # Check for delegations FIRST (before final answer)
                delegation_result = None
                if "DELEGATE_SEARCH:" in content:
                    task = content.split("DELEGATE_SEARCH:", 1)[1].strip()
                    print(f"\n🔍 DELEGATING TO SEARCH AGENT: {task}")
                    delegation_result = self._delegate_to_agent(task, "search")
                    delegations_made += 1

                elif "DELEGATE_REASONING:" in content:
                    task = content.split("DELEGATE_REASONING:", 1)[1].strip()
                    print(f"\n🧠 DELEGATING TO REASONING AGENT: {task}")
                    delegation_result = self._delegate_to_agent(task, "reasoning")
                    delegations_made += 1

                # Check for final answer ONLY if no delegation was made
                elif content and "FINAL_ANSWER:" in content:
                    result = content.split("FINAL_ANSWER:", 1)[1].strip()
                    return self._create_success_response(
                        result,
                        {"turns_used": turn + 1, "delegations_made": delegations_made},
                    )

                # Add orchestrator response to conversation
                messages.append({"role": "assistant", "content": content})

                # Add delegation result if any
                if delegation_result:
                    if delegation_result.success:
                        print(f"✅ AGENT RESULT: {delegation_result.result}")
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Agent result: {delegation_result.result}",
                            }
                        )
                    else:
                        print(f"❌ AGENT ERROR: {delegation_result.error}")
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Agent encountered an error: {delegation_result.error}",
                            }
                        )
                # If no delegation, continue conversation
                elif not any(
                    keyword in content
                    for keyword in [
                        "DELEGATE_SEARCH:",
                        "DELEGATE_REASONING:",
                        "FINAL_ANSWER:",
                    ]
                ):
                    # Orchestrator is thinking/planning, continue
                    continue

            return self._create_error_response(
                "Orchestration timeout - could not complete within turn limit",
                {"turns_used": self.max_turns, "delegations_made": delegations_made},
            )

        except Exception as e:
            return self._create_error_response(f"Orchestration failed: {str(e)}")
