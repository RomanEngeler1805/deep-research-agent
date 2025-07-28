import inspect
import importlib
from typing import Dict, Any, Callable, List
from tools import actions
import json


def get_function_schema(func: Callable) -> Dict[str, Any]:
    """
    Generate a function schema for OpenAI function calling.

    Args:
        func: The function to generate schema for

    Returns:
        Dictionary containing the function schema
    """
    sig = inspect.signature(func)

    # Get parameter information
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":  # Skip self parameter
            continue

        param_type = (
            param.annotation if param.annotation != inspect.Parameter.empty else str
        )
        param_type_str = (
            str(param_type)
            .replace("typing.", "")
            .replace("<class '", "")
            .replace("'>", "")
        )

        # Map Python types to JSON schema types
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }

        json_type = type_mapping.get(param_type_str, "string")

        properties[param_name] = {
            "type": json_type,
            "description": f"Parameter {param_name}",
        }

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",  # Add the required type field
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"Function {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def discover_tools(module_name: str = "tools.actions") -> List[Dict[str, Any]]:
    """
    Discover all functions in the specified module and return their schemas.

    Args:
        module_name: Name of the module to scan (default: "tools.actions")

    Returns:
        List of function schemas for OpenAI function calling
    """
    # Import the module
    module = importlib.import_module(module_name)

    # Functions to exclude from being available as tools
    excluded_functions = {
        "load_dotenv",  # Environment variable loading
        "_eval",  # Internal helper functions
        "eval_expr",  # Internal helper functions
        "extract_urls_from_search_results",  # Internal helper
    }

    tools = []

    # Get all functions from the module
    for name, obj in inspect.getmembers(module):
        # Only include functions (not classes, modules, etc.)
        # Exclude private functions (starting with _) and excluded functions
        if (
            inspect.isfunction(obj)
            and not name.startswith("_")
            and name not in excluded_functions
        ):

            schema = get_function_schema(obj)
            tools.append(schema)

    return tools


def get_tool_function(tool_name: str) -> Callable:
    """
    Get a function by name from the actions module.

    Args:
        tool_name: Name of the function to retrieve

    Returns:
        The function object
    """
    return getattr(actions, tool_name, None)


def execute_tool(tool_name: str, *args, **kwargs) -> Any:
    """
    Execute a tool function by name.

    Args:
        tool_name: Name of the function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function execution
    """
    func = get_tool_function(tool_name)
    if func is None:
        return f"Error: Tool '{tool_name}' not found."

    try:
        return func(*args, **kwargs)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
