from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import random

# Import the custom LLM from llm.py
from llm import CustomLLMWithTools, OutputMode


# ============================================================================
# DOCUMENT READING TOOLS
# ============================================================================

@tool
def read_document_chunk(start_line: int = None) -> str:
    """
    Read a random chunk of 50 lines from input.md file
    Returns a chunk of text from the document with line numbers
    ---
    int start_line: Optional starting line number (if not provided, reads from random position)
    """
    try:
        with open('input.md', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        if start_line is not None and 0 <= start_line < total_lines:
            start = start_line
        else:
            start = random.randint(0, max(0, total_lines - 50))
        
        end = min(start + 50, total_lines)
        chunk_lines = lines[start:end]
        
        formatted_chunk = f"=== DOCUMENT CHUNK (Lines {start+1} to {end}) ===\n"
        formatted_chunk += f"Total document lines: {total_lines}\n"
        formatted_chunk += "=" * 80 + "\n"
        
        for i, line in enumerate(chunk_lines, start=start+1):
            formatted_chunk += f"{i:4d} | {line}"
        
        formatted_chunk += "\n" + "=" * 80
        formatted_chunk += f"\nChunk contains lines {start+1}-{end} of {total_lines} total lines"
        
        print(f"[TOOL] Read document chunk: lines {start+1} to {end}")
        return formatted_chunk
        
    except FileNotFoundError:
        return "ERROR: input.md file not found in current directory!"
    except Exception as e:
        return f"ERROR reading file: {str(e)}"


@tool
def search_document_section(keyword: str, start_line: int = None) -> str:
    """
    Search for a keyword in a section of the document and return context around matches
    Returns lines containing the keyword with context, or a message if not found
    ---
    str keyword: The keyword or phrase to search for
    int start_line: Optional starting line to search from (if not provided, searches random section)
    """
    try:
        with open('input.md', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        if start_line is not None and 0 <= start_line < total_lines:
            start = start_line
        else:
            start = random.randint(0, max(0, total_lines - 50))
        
        end = min(start + 50, total_lines)
        search_lines = lines[start:end]
        
        matches = []
        for i, line in enumerate(search_lines, start=start+1):
            if keyword.lower() in line.lower():
                context_start = max(start, i - 3)
                context_end = min(end, i + 3)
                context = lines[context_start:context_end]
                
                match_text = f"\n--- Match at line {i} ---\n"
                for j, ctx_line in enumerate(context, start=context_start+1):
                    prefix = ">>> " if j == i else "    "
                    match_text += f"{prefix}{j:4d} | {ctx_line}"
                matches.append(match_text)
        
        if matches:
            result = f"Found {len(matches)} match(es) for '{keyword}' in lines {start+1}-{end}:\n"
            result += "\n".join(matches)
            print(f"[TOOL] Search found {len(matches)} matches for '{keyword}'")
        else:
            result = f"No matches for '{keyword}' found in lines {start+1}-{end} of {total_lines}. Try searching another section."
            print(f"[TOOL] No matches for '{keyword}' in lines {start+1}-{end}")
        
        return result
        
    except FileNotFoundError:
        return "ERROR: input.md file not found!"
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================================================================
# AGENT STATE AND NODES
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration_count: int


def agent_node(state: AgentState) -> AgentState:
    """The main agent that reasons and decides which tools to call"""
    messages = state["messages"]
    iteration = state.get("iteration_count", 0) + 1
    
    print(f"\n{'='*80}")
    print(f"[AGENT] Iteration {iteration}")
    print(f"[AGENT] Message history size: {len(messages)} messages")
    print(f"{'='*80}")
    
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": response,
        "iteration_count": iteration
    }


def should_continue(state: AgentState) -> str:
    """Determine if we should continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"[ROUTER] Tool calls detected -> routing to tool_node")
        return "continue"
    
    print(f"[ROUTER] No tool calls -> end")
    return "end"


# ============================================================================
# SETUP AND EXECUTION
# ============================================================================

# Configuration - Change these to customize behavior
OUTPUT_MODE = OutputMode.LANGCHAIN  # Options: MANUAL, LANGCHAIN, OPENAI_JSON
BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "gpt-oss"  # or "gpt-oss" for your local model
TEMPERATURE = 0.7

# Define tools
tools = [read_document_chunk, search_document_section]

# Create custom LLM with tool calling capability
print(f"[SETUP] Initializing Custom LLM with {OUTPUT_MODE} mode...")
llm_with_tools = CustomLLMWithTools(
    mode=OUTPUT_MODE,
    base_url=BASE_URL,
    model_name=MODEL_NAME,
    tools=tools
)

print(f"[SETUP] Registered {len(tools)} tools")
print(f"[SETUP] Using base URL: {BASE_URL}")
print(f"[SETUP] Model: {MODEL_NAME}")

# Create tool node
tool_node = ToolNode(tools)

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

print("[SETUP] LangGraph workflow compiled successfully\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    query = """Please search through the input.md document and find the definition of "Power Systems". 

The document is large, so you'll need to read through it in chunks. Keep searching different sections 
until you find where "Power Systems" is defined or described. Once you find it, provide me with the 
complete definition and the line number where you found it.

CRITICAL:
- Don't return empty response. YOU ARE NOT ALLOWED TO RETURN EMPTY RESPONSES.
- Search multiple sections if needed
- Provide the exact line number where you found the definition
- YOU ARE NOT ALLOWED TO GIVE UP WITHOUT FINDING THE DEFINITION
"""
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "iteration_count": 0
    }
    
    print("="*80)
    print("SEARCH QUERY:")
    print("="*80)
    print(query)
    print("="*80)
    print(f"\nSTARTING AGENT EXECUTION (Mode: {OUTPUT_MODE})...\n")
    
    try:
        final_state = app.invoke(initial_state)
        
        print("\n" + "="*80)
        print("EXECUTION COMPLETE")
        print("="*80)
        print(f"Total iterations: {final_state['iteration_count']}")
        
        print("\n" + "="*80)
        print("FINAL ANSWER:")
        print("="*80)
        
        # Get the last AI message
        ai_messages = [msg for msg in final_state['messages'] if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1]
            if last_ai_message.content:
                print(last_ai_message.content)
            else:
                print("[WARNING] Final AI message has empty content")
                print(f"Last message object: {last_ai_message}")
                
                # Check if there are tool calls
                if hasattr(last_ai_message, 'tool_calls') and last_ai_message.tool_calls:
                    print(f"\nNote: Message contains tool calls: {last_ai_message.tool_calls}")
        else:
            print("[ERROR] No AI messages found in final state")
            print(f"Total messages in final state: {len(final_state['messages'])}")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Agent execution failed: {e}")
        import traceback
        traceback.print_exc()