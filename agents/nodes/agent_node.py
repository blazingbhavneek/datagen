"""
Custom LLM wrapper with tool calling using three different structured output modes.
Supports: LangChain structured output, OpenAI JSON schema, and manual JSON parsing.
NOW WITH ASYNC SUPPORT.
"""

import asyncio
import inspect
import json
import random
from enum import Enum
from typing import Any, Callable, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model

# System prompts (configurable but with defaults)
PLANNER_SYSTEM_PROMPT = """You are an AI planning assistant. Your job is to:
1. Analyze the conversation history
2. Decide if a tool call is needed or if you can provide a final answer
3. If calling a tool, provide a detailed natural language description of what information you need

IMPORTANT:
- Provide clear, detailed descriptions of what you want from the tool
- Explain parameter values in natural language (the executor will convert them)
- If you've already used certain parameters (e.g., searched specific sections), choose DIFFERENT ones
- Be specific and thorough in your descriptions
- The executor LLM will rely on your description to generate parameters

OUTPUT:
- content: Your final answer (if no tool needed) OR detailed description of what you want from the tool
- tool_name: null (if final answer) or the exact tool name (if calling tool)"""


EXECUTOR_SYSTEM_PROMPT = """You are a parameter generation assistant. Your job is to:
1. Read the planner's description of what information is needed
2. Convert that description into exact parameter values matching the tool's schema
3. Generate valid JSON that matches the required structure

IMPORTANT:
- Follow the planner's instructions exactly
- Use appropriate parameter values based on the description
- Ensure all required parameters are provided
- Generate ONLY valid JSON, no other text
- If the planner's description is unclear, do your best to infer reasonable values"""


DOCSTRING_FORMAT_GUIDE = """
Tool docstring must follow this format:

'''
Description line 1
Description line 2 (optional, multiple lines allowed)
---
<type> <param_name>: Parameter description
<type> <param_name>: Parameter description
'''

Example:
'''
Search for content in a file within a specific line range
Returns the matching lines
---
str file_path: Path to the file to search
int start_line: Starting line number (1-indexed)
int end_line: Ending line number (inclusive)
str query: Search term to find
'''
"""


class OutputMode(str, Enum):
    """Available structured output modes"""

    LANGCHAIN = "langchain"
    OPENAI_JSON = "openai_json"
    MANUAL = "manual"


class ToolCallDecision(BaseModel):
    """Simplified planner decision model"""

    content: str = Field(
        ...,
        description="Final answer or detailed description of needed tool information",
    )
    tool_name: Optional[str] = Field(
        None, description="Name of tool to call, or null for final answer"
    )


class CustomLLMWithTools:
    """
    Custom LLM wrapper implementing tool calling with structured outputs.
    Two-stage process:
    1. Planning LLM: Decides tool + describes what information is needed
    2. Executor LLM: Converts description to structured parameters
    
    Supports both sync and async tools and invocation.
    """

    def __init__(
        self,
        mode: OutputMode = OutputMode.MANUAL,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "gpt-oss",
        tools: List[Callable] = None,
        planner_system_prompt: str = PLANNER_SYSTEM_PROMPT,
        executor_system_prompt: str = EXECUTOR_SYSTEM_PROMPT,
    ):
        """
        Initialize the custom LLM wrapper

        Args:
            mode: Structured output mode (langchain, openai_json, or manual)
            base_url: Base URL for the LLM API
            model_name: Model name to use
            tools: List of tool functions to register (sync or async)
            planner_system_prompt: System prompt for planning LLM
            executor_system_prompt: System prompt for executor LLM
        """
        self.mode = mode
        self.planner_system_prompt = planner_system_prompt
        self.executor_system_prompt = executor_system_prompt

        # Initialize LLM clients based on mode
        if mode == OutputMode.LANGCHAIN:
            from langchain_openai import ChatOpenAI

            self.planner_llm = ChatOpenAI(base_url=base_url, model=model_name)
            self.executor_llm = ChatOpenAI(base_url=base_url, model=model_name)

        elif mode == OutputMode.OPENAI_JSON:
            from openai import OpenAI

            self.client = OpenAI(base_url=base_url, api_key="dummy")
            self.model_name = model_name

        elif mode == OutputMode.MANUAL:
            from langchain_openai import ChatOpenAI

            self.planner_llm = ChatOpenAI(base_url=base_url, model=model_name)
            self.executor_llm = ChatOpenAI(base_url=base_url, model=model_name)

        self.tools = {}
        self.tool_schemas = {}

        if tools:
            for tool_func in tools:
                self._register_tool(tool_func)

    def _get_tool_function(self, tool_func: Callable) -> Callable:
        """Extract the actual function from a tool wrapper"""
        # Try multiple attributes in order
        for attr in ['func', 'coroutine', '_target', '__wrapped__']:
            if hasattr(tool_func, attr):
                result = getattr(tool_func, attr)
                if result is not None and callable(result):
                    return result
        
        # If it's already callable, return it
        if callable(tool_func):
            return tool_func
            
        # Last resort: check if it has invoke/ainvoke methods (LangChain pattern)
        if hasattr(tool_func, 'invoke') or hasattr(tool_func, 'ainvoke'):
            # For LangChain tools, inspect the args_schema instead
            if hasattr(tool_func, 'args_schema'):
                # We'll handle this differently - extract from schema
                return None  # Signal to use schema-based approach
        
        raise ValueError(f"Could not extract callable from tool: {tool_func}")

    def _parse_docstring(self, docstring: str) -> tuple[str, dict]:
        """
        Parse tool docstring in the required format.

        Returns:
            (description, parameters_dict)

        Raises:
            ValueError: If docstring doesn't match required format
        """
        if not docstring or "---" not in docstring:
            raise ValueError(
                f"Tool docstring must contain '---' separator.\n{DOCSTRING_FORMAT_GUIDE}"
            )

        parts = docstring.split("---", 1)
        description = parts[0].strip()
        params_section = parts[1].strip()

        if not description:
            raise ValueError(
                f"Tool description cannot be empty.\n{DOCSTRING_FORMAT_GUIDE}"
            )

        params_dict = {}
        if params_section:
            for line in params_section.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Parse: <type> <param_name>: Description
                try:
                    type_and_name, desc = line.split(":", 1)
                    type_str, param_name = type_and_name.strip().split(None, 1)

                    # Convert type string to actual type
                    type_map = {
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                    }
                    param_type = type_map.get(type_str.lower(), str)

                    params_dict[param_name] = {
                        "type": param_type,
                        "description": desc.strip(),
                    }
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse parameter line: '{line}'\n"
                        f"Error: {e}\n{DOCSTRING_FORMAT_GUIDE}"
                    )

        return description, params_dict

    def _register_tool(self, tool_func: Callable):
        """Register a tool and extract its schema from docstring"""
        tool_name = tool_func.name if hasattr(tool_func, "name") else tool_func.__name__
        self.tools[tool_name] = tool_func

        # Get docstring
        doc = (
            tool_func.description
            if hasattr(tool_func, "description")
            else (tool_func.__doc__ or "")
        )

        # Parse docstring
        try:
            description, params_from_doc = self._parse_docstring(doc)
        except ValueError as e:
            raise ValueError(f"Error in tool '{tool_name}': {e}")

        # Get the actual function to inspect
        actual_func = self._get_tool_function(tool_func)
        
        # If we couldn't unwrap it, try using LangChain's args_schema
        if actual_func is None:
            if hasattr(tool_func, 'args_schema') and tool_func.args_schema is not None:
                # Use Pydantic schema from LangChain tool
                schema_fields = tool_func.args_schema.model_fields
                params_info = {}
                
                for param_name, field_info in schema_fields.items():
                    if param_name not in params_from_doc:
                        raise ValueError(
                            f"Tool '{tool_name}': Parameter '{param_name}' in schema "
                            f"but not documented in docstring.\n{DOCSTRING_FORMAT_GUIDE}"
                        )
                    
                    params_info[param_name] = {
                        "type": params_from_doc[param_name]["type"],
                        "description": params_from_doc[param_name]["description"],
                        "default": field_info.default if field_info.default is not None else None,
                        "required": field_info.is_required(),
                    }
                
                self.tool_schemas[tool_name] = {
                    "name": tool_name,
                    "description": description,
                    "parameters": params_info,
                }
                
                print(f"[REGISTERED] Tool '{tool_name}' with {len(params_info)} parameters (from schema)")
                return
            else:
                raise ValueError(f"Could not extract function or schema from tool '{tool_name}'")
        
        # Get function signature (works for both sync and async)
        sig = inspect.signature(actual_func)

        # Build parameter schema
        params_info = {}
        for param_name, param in sig.parameters.items():
            if param_name not in params_from_doc:
                raise ValueError(
                    f"Tool '{tool_name}': Parameter '{param_name}' in signature "
                    f"but not documented in docstring.\n{DOCSTRING_FORMAT_GUIDE}"
                )

            param_type = params_from_doc[param_name]["type"]
            param_desc = params_from_doc[param_name]["description"]
            param_default = (
                param.default if param.default != inspect.Parameter.empty else None
            )

            params_info[param_name] = {
                "type": param_type,
                "description": param_desc,
                "default": param_default,
                "required": param.default == inspect.Parameter.empty,
            }

        # Verify all documented params are in signature
        sig_params = set(sig.parameters.keys())
        doc_params = set(params_from_doc.keys())
        extra_params = doc_params - sig_params
        if extra_params:
            raise ValueError(
                f"Tool '{tool_name}': Parameters {extra_params} documented "
                f"but not in function signature.\n{DOCSTRING_FORMAT_GUIDE}"
            )

        self.tool_schemas[tool_name] = {
            "name": tool_name,
            "description": description,
            "parameters": params_info,
        }

        print(f"[REGISTERED] Tool '{tool_name}' with {len(params_info)} parameters")

    def _create_tool_param_model(self, tool_name: str) -> type[BaseModel]:
        """Dynamically create a Pydantic model for tool parameters"""
        if tool_name not in self.tool_schemas:
            raise ValueError(f"Unknown tool: {tool_name}")

        schema = self.tool_schemas[tool_name]
        fields = {}

        for param_name, param_info in schema["parameters"].items():
            param_type = param_info["type"]
            param_desc = param_info["description"]
            param_default = param_info["default"]

            if not param_info["required"]:
                field_type = Optional[param_type]
                fields[param_name] = (
                    field_type,
                    Field(default=param_default, description=param_desc),
                )
            else:
                fields[param_name] = (param_type, Field(..., description=param_desc))

        return create_model(f"{tool_name}_params", **fields)

    def _format_tools_for_planning(self) -> str:
        """Format tool descriptions for the planning LLM"""
        if not self.tool_schemas:
            return "No tools available."

        descriptions = ["AVAILABLE TOOLS:\n"]
        for tool_name, schema in self.tool_schemas.items():
            descriptions.append(f"Tool: {tool_name}")
            descriptions.append(f"Description: {schema['description']}")
            descriptions.append("Parameters:")
            for param_name, param_info in schema["parameters"].items():
                req = (
                    "REQUIRED"
                    if param_info["required"]
                    else f"optional (default: {param_info['default']})"
                )
                descriptions.append(
                    f"  - {param_name} ({param_info['type'].__name__}): "
                    f"{param_info['description']} [{req}]"
                )
            descriptions.append("")

        return "\n".join(descriptions)

    def _invoke_planner_langchain(
        self, messages: List[BaseMessage], tools_desc: str
    ) -> ToolCallDecision:
        """Invoke planner using LangChain structured output"""
        from pydantic import field_validator

        # Create dynamic model with runtime validation
        tool_names = list(self.tool_schemas.keys())

        class PlannerDecision(BaseModel):
            content: str = Field(..., min_length=1)
            tool_name: Optional[str] = None

            @field_validator("tool_name")
            @classmethod
            def validate_tool_name(cls, v):
                if v is not None and v not in tool_names:
                    raise ValueError(
                        f"Invalid tool name: {v}. Must be one of {tool_names}"
                    )
                return v

        structured_llm = self.planner_llm.with_structured_output(PlannerDecision)

        planner_msg = HumanMessage(content=f"{tools_desc}\n\nAnalyze and decide.")
        response = structured_llm.invoke(
            [HumanMessage(content=self.planner_system_prompt)]
            + messages
            + [planner_msg]
        )

        return ToolCallDecision(content=response.content, tool_name=response.tool_name)

    async def _invoke_planner_langchain_async(
        self, messages: List[BaseMessage], tools_desc: str
    ) -> ToolCallDecision:
        """Async invoke planner using LangChain structured output"""
        from pydantic import field_validator

        tool_names = list(self.tool_schemas.keys())

        class PlannerDecision(BaseModel):
            content: str = Field(..., min_length=1)
            tool_name: Optional[str] = None

            @field_validator("tool_name")
            @classmethod
            def validate_tool_name(cls, v):
                if v is not None and v not in tool_names:
                    raise ValueError(
                        f"Invalid tool name: {v}. Must be one of {tool_names}"
                    )
                return v

        structured_llm = self.planner_llm.with_structured_output(PlannerDecision)

        planner_msg = HumanMessage(content=f"{tools_desc}\n\nAnalyze and decide.")
        response = await structured_llm.ainvoke(
            [HumanMessage(content=self.planner_system_prompt)]
            + messages
            + [planner_msg]
        )

        return ToolCallDecision(content=response.content, tool_name=response.tool_name)

    def _invoke_planner_openai(
        self, messages: List[BaseMessage], tools_desc: str
    ) -> ToolCallDecision:
        """Invoke planner using OpenAI JSON schema"""
        # Create schema
        schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "minLength": 1},
                "tool_name": {
                    "type": ["string", "null"],
                    "enum": list(self.tool_schemas.keys()) + [None],
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        }

        # Convert messages to OpenAI format
        openai_messages = [{"role": "system", "content": self.planner_system_prompt}]
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            openai_messages.append({"role": role, "content": msg.content})

        openai_messages.append(
            {"role": "user", "content": f"{tools_desc}\n\nAnalyze and decide."}
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "planner_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        data = json.loads(response.choices[0].message.content)
        return ToolCallDecision(**data)

    async def _invoke_planner_openai_async(
        self, messages: List[BaseMessage], tools_desc: str
    ) -> ToolCallDecision:
        """Async invoke planner using OpenAI JSON schema"""
        from openai import AsyncOpenAI
        
        # Create async client if needed
        if not hasattr(self, '_async_client'):
            self._async_client = AsyncOpenAI(base_url=self.client.base_url, api_key="dummy")
        
        schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string", "minLength": 1},
                "tool_name": {
                    "type": ["string", "null"],
                    "enum": list(self.tool_schemas.keys()) + [None],
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        }

        openai_messages = [{"role": "system", "content": self.planner_system_prompt}]
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            openai_messages.append({"role": role, "content": msg.content})

        openai_messages.append(
            {"role": "user", "content": f"{tools_desc}\n\nAnalyze and decide."}
        )

        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "planner_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        data = json.loads(response.choices[0].message.content)
        return ToolCallDecision(**data)

    def _invoke_planner_manual(
        self, messages: List[BaseMessage], tools_desc: str
    ) -> ToolCallDecision:
        """Invoke planner with manual JSON parsing"""
        prompt = f"""{tools_desc}

OUTPUT FORMAT (JSON only):
{{
    "content": "your final answer OR detailed description of tool information needed",
    "tool_name": "tool_name" or null
}}

IMPORTANT:
- content must not be empty
- If calling a tool, describe in detail what information you need
- Respond with ONLY the JSON object"""

        planner_messages = (
            [HumanMessage(content=self.planner_system_prompt)]
            + messages
            + [HumanMessage(content=prompt)]
        )

        max_retries = 3
        for attempt in range(max_retries):
            response = self.planner_llm.invoke(planner_messages)

            try:
                content = response.content.strip()

                # Extract JSON
                if content.startswith("```"):
                    import re

                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
                    )
                    if json_match:
                        content = json_match.group(1)

                data = json.loads(content)

                # Validate
                if not data.get("content") or not data["content"].strip():
                    raise ValueError("content field is empty or missing")

                return ToolCallDecision(**data)

            except Exception as e:
                if attempt < max_retries - 1:
                    planner_messages.append(AIMessage(content=response.content))
                    planner_messages.append(
                        HumanMessage(
                            content=f"ERROR: {e}\n\nProvide valid JSON with non-empty 'content' field."
                        )
                    )
                else:
                    raise ValueError(
                        f"Planner failed after {max_retries} attempts: {e}"
                    )

    async def _invoke_planner_manual_async(
        self, messages: List[BaseMessage], tools_desc: str
    ) -> ToolCallDecision:
        """Async invoke planner with manual JSON parsing"""
        prompt = f"""{tools_desc}

OUTPUT FORMAT (JSON only):
{{
    "content": "your final answer OR detailed description of tool information needed",
    "tool_name": "tool_name" or null
}}

IMPORTANT:
- content must not be empty
- If calling a tool, describe in detail what information you need
- Respond with ONLY the JSON object"""

        planner_messages = (
            [HumanMessage(content=self.planner_system_prompt)]
            + messages
            + [HumanMessage(content=prompt)]
        )

        max_retries = 3
        for attempt in range(max_retries):
            response = await self.planner_llm.ainvoke(planner_messages)

            try:
                content = response.content.strip()

                if content.startswith("```"):
                    import re
                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
                    )
                    if json_match:
                        content = json_match.group(1)

                data = json.loads(content)

                if not data.get("content") or not data["content"].strip():
                    raise ValueError("content field is empty or missing")

                return ToolCallDecision(**data)

            except Exception as e:
                if attempt < max_retries - 1:
                    planner_messages.append(AIMessage(content=response.content))
                    planner_messages.append(
                        HumanMessage(
                            content=f"ERROR: {e}\n\nProvide valid JSON with non-empty 'content' field."
                        )
                    )
                else:
                    raise ValueError(
                        f"Planner failed after {max_retries} attempts: {e}"
                    )

    def _invoke_executor_langchain(
        self, tool_name: str, planner_content: str, ParamModel: type[BaseModel]
    ) -> BaseModel:
        """Invoke executor using LangChain structured output"""
        schema = self.tool_schemas[tool_name]

        structured_llm = self.executor_llm.with_structured_output(ParamModel)

        prompt = f"""TOOL: {tool_name}
DESCRIPTION: {schema['description']}

PLANNER'S REQUEST:
{planner_content}

Generate the parameters to fulfill this request."""

        response = structured_llm.invoke(
            [
                HumanMessage(content=self.executor_system_prompt),
                HumanMessage(content=prompt),
            ]
        )

        return response

    async def _invoke_executor_langchain_async(
        self, tool_name: str, planner_content: str, ParamModel: type[BaseModel]
    ) -> BaseModel:
        """Async invoke executor using LangChain structured output"""
        schema = self.tool_schemas[tool_name]

        structured_llm = self.executor_llm.with_structured_output(ParamModel)

        prompt = f"""TOOL: {tool_name}
DESCRIPTION: {schema['description']}

PLANNER'S REQUEST:
{planner_content}

Generate the parameters to fulfill this request."""

        response = await structured_llm.ainvoke(
            [
                HumanMessage(content=self.executor_system_prompt),
                HumanMessage(content=prompt),
            ]
        )

        return response

    def _invoke_executor_openai(
        self, tool_name: str, planner_content: str, ParamModel: type[BaseModel]
    ) -> BaseModel:
        """Invoke executor using OpenAI JSON schema"""
        schema = self.tool_schemas[tool_name]

        # Build JSON schema from Pydantic model
        pydantic_schema = ParamModel.model_json_schema()
        openai_schema = {
            "type": "object",
            "properties": pydantic_schema["properties"],
            "required": pydantic_schema.get("required", []),
            "additionalProperties": False,
        }

        prompt = f"""TOOL: {tool_name}
DESCRIPTION: {schema['description']}

PLANNER'S REQUEST:
{planner_content}

Generate the parameters."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.executor_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": f"{tool_name}_params",
                    "strict": True,
                    "schema": openai_schema,
                },
            },
        )

        data = json.loads(response.choices[0].message.content)
        return ParamModel(**data)

    async def _invoke_executor_openai_async(
        self, tool_name: str, planner_content: str, ParamModel: type[BaseModel]
    ) -> BaseModel:
        """Async invoke executor using OpenAI JSON schema"""
        if not hasattr(self, '_async_client'):
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(base_url=self.client.base_url, api_key="dummy")
        
        schema = self.tool_schemas[tool_name]

        pydantic_schema = ParamModel.model_json_schema()
        openai_schema = {
            "type": "object",
            "properties": pydantic_schema["properties"],
            "required": pydantic_schema.get("required", []),
            "additionalProperties": False,
        }

        prompt = f"""TOOL: {tool_name}
DESCRIPTION: {schema['description']}

PLANNER'S REQUEST:
{planner_content}

Generate the parameters."""

        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.executor_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": f"{tool_name}_params",
                    "strict": True,
                    "schema": openai_schema,
                },
            },
        )

        data = json.loads(response.choices[0].message.content)
        return ParamModel(**data)

    def _invoke_executor_manual(
        self, tool_name: str, planner_content: str, ParamModel: type[BaseModel]
    ) -> BaseModel:
        """Invoke executor with manual JSON parsing"""
        schema = self.tool_schemas[tool_name]

        # Build expected JSON structure
        json_structure = "{\n"
        for param, info in schema["parameters"].items():
            json_structure += f'    "{param}": <{info["type"].__name__}>,\n'
        json_structure = json_structure.rstrip(",\n") + "\n}"

        prompt = f"""TOOL: {tool_name}
DESCRIPTION: {schema['description']}

PLANNER'S REQUEST:
{planner_content}

REQUIRED JSON FORMAT:
{json_structure}

Generate ONLY the JSON object with appropriate parameter values."""

        max_retries = 3
        messages = [
            HumanMessage(content=self.executor_system_prompt),
            HumanMessage(content=prompt),
        ]

        for attempt in range(max_retries):
            response = self.executor_llm.invoke(messages)

            try:
                content = response.content.strip()

                # Extract JSON
                if content.startswith("```"):
                    import re

                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
                    )
                    if json_match:
                        content = json_match.group(1)

                data = json.loads(content)
                return ParamModel(**data)

            except Exception as e:
                if attempt < max_retries - 1:
                    messages.append(AIMessage(content=response.content))
                    messages.append(
                        HumanMessage(
                            content=f"VALIDATION ERROR: {e}\n\nGenerate valid JSON matching the schema."
                        )
                    )
                else:
                    raise ValueError(
                        f"Executor failed after {max_retries} attempts: {e}"
                    )

    async def _invoke_executor_manual_async(
        self, tool_name: str, planner_content: str, ParamModel: type[BaseModel]
    ) -> BaseModel:
        """Async invoke executor with manual JSON parsing"""
        schema = self.tool_schemas[tool_name]

        json_structure = "{\n"
        for param, info in schema["parameters"].items():
            json_structure += f'    "{param}": <{info["type"].__name__}>,\n'
        json_structure = json_structure.rstrip(",\n") + "\n}"

        prompt = f"""TOOL: {tool_name}
DESCRIPTION: {schema['description']}

PLANNER'S REQUEST:
{planner_content}

REQUIRED JSON FORMAT:
{json_structure}

Generate ONLY the JSON object with appropriate parameter values."""

        max_retries = 3
        messages = [
            HumanMessage(content=self.executor_system_prompt),
            HumanMessage(content=prompt),
        ]

        for attempt in range(max_retries):
            response = await self.executor_llm.ainvoke(messages)

            try:
                content = response.content.strip()

                if content.startswith("```"):
                    import re
                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
                    )
                    if json_match:
                        content = json_match.group(1)

                data = json.loads(content)
                return ParamModel(**data)

            except Exception as e:
                if attempt < max_retries - 1:
                    messages.append(AIMessage(content=response.content))
                    messages.append(
                        HumanMessage(
                            content=f"VALIDATION ERROR: {e}\n\nGenerate valid JSON matching the schema."
                        )
                    )
                else:
                    raise ValueError(
                        f"Executor failed after {max_retries} attempts: {e}"
                    )

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Main invoke method - coordinates planner and executor (synchronous)
        """
        tools_desc = self._format_tools_for_planning()

        # ===== STAGE 1: PLANNING =====
        print(f"\n[{self.mode.upper()}] Stage 1: Planning...")

        if self.mode == OutputMode.LANGCHAIN:
            plan = self._invoke_planner_langchain(messages, tools_desc)
        elif self.mode == OutputMode.OPENAI_JSON:
            plan = self._invoke_planner_openai(messages, tools_desc)
        else:  # MANUAL
            plan = self._invoke_planner_manual(messages, tools_desc)

        print(f"[{self.mode.upper()}] Plan: tool={plan.tool_name}")
        print(f"[{self.mode.upper()}] Content: {plan.content[:100]}...")

        # If no tool needed, return final answer
        if not plan.tool_name:
            print(f"[{self.mode.upper()}] No tool needed, returning final answer")
            return AIMessage(content=plan.content)

        # Validate tool exists
        if plan.tool_name not in self.tools:
            return AIMessage(
                content=f"Error: Unknown tool '{plan.tool_name}'. Available: {list(self.tools.keys())}"
            )

        # ===== STAGE 2: PARAMETER GENERATION =====
        print(
            f"[{self.mode.upper()}] Stage 2: Generating parameters for '{plan.tool_name}'..."
        )

        ParamModel = self._create_tool_param_model(plan.tool_name)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.mode == OutputMode.LANGCHAIN:
                    params = self._invoke_executor_langchain(
                        plan.tool_name, plan.content, ParamModel
                    )
                elif self.mode == OutputMode.OPENAI_JSON:
                    params = self._invoke_executor_openai(
                        plan.tool_name, plan.content, ParamModel
                    )
                else:  # MANUAL
                    params = self._invoke_executor_manual(
                        plan.tool_name, plan.content, ParamModel
                    )

                print(
                    f"[{self.mode.upper()}] Generated parameters: {params.model_dump()}"
                )
                break

            except Exception as e:
                print(
                    f"[{self.mode.upper()}] Executor error (attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    error_msg = f"""The executor failed to generate valid parameters from your description.

ERROR: {e}

Please provide a MORE DETAILED and CLEARER description of the tool parameters needed.
Be specific about values, formats, and requirements."""

                    print(
                        f"[{self.mode.upper()}] Asking planner for better description..."
                    )

                    retry_messages = messages + [
                        AIMessage(content=plan.content),
                        HumanMessage(content=error_msg),
                    ]

                    if self.mode == OutputMode.LANGCHAIN:
                        plan = self._invoke_planner_langchain(
                            retry_messages, tools_desc
                        )
                    elif self.mode == OutputMode.OPENAI_JSON:
                        plan = self._invoke_planner_openai(retry_messages, tools_desc)
                    else:
                        plan = self._invoke_planner_manual(retry_messages, tools_desc)

                    print(f"[{self.mode.upper()}] Revised content: {plan.content[:100]}...")
                else:
                    return AIMessage(
                        content=f"Error: Failed to generate parameters after {max_retries} attempts: {e}"
                    )

        # Create tool call
        from langchain_core.messages.tool import ToolCall

        tool_call_id = f"call_{random.randint(1000, 9999)}"
        tool_call = ToolCall(
            name=plan.tool_name, args=params.model_dump(), id=tool_call_id
        )

        print(f"[{self.mode.upper()}] Tool call created successfully\n")
        return AIMessage(content="", tool_calls=[tool_call])

    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Main async invoke method - coordinates planner and executor
        """
        tools_desc = self._format_tools_for_planning()

        # ===== STAGE 1: PLANNING =====
        print(f"\n[{self.mode.upper()}] Stage 1: Planning (async)...")

        if self.mode == OutputMode.LANGCHAIN:
            plan = await self._invoke_planner_langchain_async(messages, tools_desc)
        elif self.mode == OutputMode.OPENAI_JSON:
            plan = await self._invoke_planner_openai_async(messages, tools_desc)
        else:  # MANUAL
            plan = await self._invoke_planner_manual_async(messages, tools_desc)

        print(f"[{self.mode.upper()}] Plan: tool={plan.tool_name}")
        print(f"[{self.mode.upper()}] Content: {plan.content[:100]}...")

        if not plan.tool_name:
            print(f"[{self.mode.upper()}] No tool needed, returning final answer")
            return AIMessage(content=plan.content)

        if plan.tool_name not in self.tools:
            return AIMessage(
                content=f"Error: Unknown tool '{plan.tool_name}'. Available: {list(self.tools.keys())}"
            )

        # ===== STAGE 2: PARAMETER GENERATION =====
        print(
            f"[{self.mode.upper()}] Stage 2: Generating parameters for '{plan.tool_name}' (async)..."
        )

        ParamModel = self._create_tool_param_model(plan.tool_name)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.mode == OutputMode.LANGCHAIN:
                    params = await self._invoke_executor_langchain_async(
                        plan.tool_name, plan.content, ParamModel
                    )
                elif self.mode == OutputMode.OPENAI_JSON:
                    params = await self._invoke_executor_openai_async(
                        plan.tool_name, plan.content, ParamModel
                    )
                else:  # MANUAL
                    params = await self._invoke_executor_manual_async(
                        plan.tool_name, plan.content, ParamModel
                    )

                print(
                    f"[{self.mode.upper()}] Generated parameters: {params.model_dump()}"
                )
                break

            except Exception as e:
                print(
                    f"[{self.mode.upper()}] Executor error (attempt {attempt + 1}/{max_retries}): {e}"
                )

                if attempt < max_retries - 1:
                    error_msg = f"""The executor failed to generate valid parameters from your description.

ERROR: {e}

Please provide a MORE DETAILED and CLEARER description of the tool parameters needed.
Be specific about values, formats, and requirements."""

                    print(
                        f"[{self.mode.upper()}] Asking planner for better description..."
                    )

                    retry_messages = messages + [
                        AIMessage(content=plan.content),
                        HumanMessage(content=error_msg),
                    ]

                    if self.mode == OutputMode.LANGCHAIN:
                        plan = await self._invoke_planner_langchain_async(
                            retry_messages, tools_desc
                        )
                    elif self.mode == OutputMode.OPENAI_JSON:
                        plan = await self._invoke_planner_openai_async(retry_messages, tools_desc)
                    else:
                        plan = await self._invoke_planner_manual_async(retry_messages, tools_desc)

                    print(f"[{self.mode.upper()}] Revised content: {plan.content[:100]}...")
                else:
                    return AIMessage(
                        content=f"Error: Failed to generate parameters after {max_retries} attempts: {e}"
                    )

        from langchain_core.messages.tool import ToolCall

        tool_call_id = f"call_{random.randint(1000, 9999)}"
        tool_call = ToolCall(
            name=plan.tool_name, args=params.model_dump(), id=tool_call_id
        )

        print(f"[{self.mode.upper()}] Tool call created successfully\n")
        return AIMessage(content="", tool_calls=[tool_call])

    def bind_tools(self, tools: List[Callable]):
        """Bind tools to this LLM (for compatibility)"""
        for tool_func in tools:
            self._register_tool(tool_func)
        return self


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from langchain_core.tools import tool

    # Define test tools - both sync and async
    @tool
    def search_file(
        file_path: str, start_line: int, end_line: int, query: str = ""
    ) -> str:
        """
        Search for content in a file within a specific line range
        Returns matching lines or all lines in range if no query provided
        ---
        str file_path: Path to the file to search in
        int start_line: Starting line number (1-indexed)
        int end_line: Ending line number (inclusive)
        str query: Optional search term to filter lines
        """
        return f"Searched {file_path} lines {start_line}-{end_line} for '{query}'"

    @tool
    async def async_calculate_sum(a: int, b: int) -> int:
        """
        Add two numbers together and return the result (async version)
        ---
        int a: First number to add
        int b: Second number to add
        """
        await asyncio.sleep(0.1)  # Simulate async work
        return a + b

    @tool
    async def async_get_weather(city: str, units: str = "celsius") -> str:
        """
        Get current weather information for a city (async version)
        Returns temperature and conditions
        ---
        str city: Name of the city to get weather for
        str units: Temperature units (celsius or fahrenheit)
        """
        await asyncio.sleep(0.1)  # Simulate async work
        return f"Weather in {city}: 22°{units[0].upper()}, sunny"

    print("=" * 80)
    print("TESTING CUSTOM LLM WITH ASYNC SUPPORT")
    print("=" * 80)

    test_tools = [search_file, async_calculate_sum, async_get_weather]

    # Test messages
    test_messages_sync = [
        HumanMessage(
            content="Search the file 'data.txt' from line 10 to 20 for the word 'error'"
        )
    ]
    
    test_messages_async = [
        HumanMessage(content="What's the sum of 42 and 18?")
    ]

    # Test sync invocation
    print(f"\n{'='*80}")
    print(f"TESTING SYNC INVOCATION (MANUAL MODE)")
    print(f"{'='*80}")

    try:
        llm = CustomLLMWithTools(
            mode=OutputMode.MANUAL,
            base_url="http://localhost:8000/v1",
            model_name="gpt-oss",
            tools=test_tools,
        )

        result = llm.invoke(test_messages_sync)
        print(f"\n[RESULT] Type: {type(result)}")
        print(f"[RESULT] Content: {result.content}")
        if result.tool_calls:
            print(f"[RESULT] Tool Calls: {len(result.tool_calls)}")
            for tc in result.tool_calls:
                print(f"  - {tc['name']}: {tc['args']}")

    except Exception as e:
        print(f"\n[ERROR] Sync test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test async invocation
    print(f"\n{'='*80}")
    print(f"TESTING ASYNC INVOCATION (MANUAL MODE)")
    print(f"{'='*80}")

    async def test_async():
        try:
            llm = CustomLLMWithTools(
                mode=OutputMode.MANUAL,
                base_url="http://localhost:8000/v1",
                model_name="gpt-oss",
                tools=test_tools,
            )

            result = await llm.ainvoke(test_messages_async)
            print(f"\n[RESULT] Type: {type(result)}")
            print(f"[RESULT] Content: {result.content}")
            if result.tool_calls:
                print(f"[RESULT] Tool Calls: {len(result.tool_calls)}")
                for tc in result.tool_calls:
                    print(f"  - {tc['name']}: {tc['args']}")

        except Exception as e:
            print(f"\n[ERROR] Async test failed: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test_async())

    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")
