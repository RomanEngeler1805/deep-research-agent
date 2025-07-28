from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    SEARCH = "search"
    REASONING = "reasoning"
    GENERAL = "general"


@dataclass
class AgentCapability:
    """Description of what an agent can do"""

    name: str
    description: str
    best_for: List[str]
    example_tasks: List[str]


@dataclass
class AgentRequest:
    """Request sent to an agent"""

    task: str
    task_type: TaskType = TaskType.GENERAL
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Response from an agent"""

    result: str
    success: bool = True
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    agent_name: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt"""
        pass

    @abstractmethod
    def get_capabilities(self) -> AgentCapability:
        """Return description of agent's capabilities"""
        pass

    @abstractmethod
    def can_handle(self, request: AgentRequest) -> bool:
        """Check if this agent can handle the request"""
        pass

    @abstractmethod
    def execute(self, request: AgentRequest) -> AgentResponse:
        """Execute the agent's task"""
        pass

    def _create_success_response(
        self, result: str, metadata: Dict[str, Any] = None
    ) -> AgentResponse:
        """Helper to create success response"""
        return AgentResponse(
            result=result, success=True, agent_name=self.name, metadata=metadata or {}
        )

    def _create_error_response(
        self, error: str, metadata: Dict[str, Any] = None
    ) -> AgentResponse:
        """Helper to create error response"""
        return AgentResponse(
            result="",
            success=False,
            error=error,
            agent_name=self.name,
            metadata=metadata or {},
        )
