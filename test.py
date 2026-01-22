from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import random

# Define the document reading tool
@tool
def read_document_chunk(start_line: int = None) -> str:
    """Read a random chunk of 50 lines from input.md file.
    
    Args:
        start_line: Optional starting line number. If not provided, will read from a random position.
    
    Returns:
        A chunk of text from the document with line numbers.
    """
    try:
        with open('input.md', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # If start_line is provided, use it; otherwise pick random
        if start_line is not None and 0 <= start_line < total_lines:
            start = start_line
        else:
            start = random.randint(0, max(0, total_lines - 50))
        
        end = min(start + 50, total_lines)
        chunk_lines = lines[start:end]
        
        # Format with line numbers
        formatted_chunk = f"=== DOCUMENT CHUNK (Lines {start+1} to {end}) ===\n"
        formatted_chunk += f"Total document lines: {total_lines}\n"
        formatted_chunk += "=" * 80 + "\n"
        
        for i, line in enumerate(chunk_lines, start=start+1):
            formatted_chunk += f"{i:4d} | {line}"
        
        formatted_chunk += "\n" + "=" * 80
        formatted_chunk += f"\nChunk contains lines {start+1}-{end} of {total_lines} total lines"
        
        print(f"[TOOL] Read document chunk: lines {start+1} to {end} ({len(chunk_lines)} lines, ~{len(formatted_chunk)} chars)")
        
        return formatted_chunk
        
    except FileNotFoundError:
        error_msg = "ERROR: input.md file not found in current directory!"
        print(f"[TOOL] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"ERROR reading file: {str(e)}"
        print(f"[TOOL] {error_msg}")
        return error_msg

@tool
def search_document_section(keyword: str, start_line: int = None) -> str:
    """Search for a keyword in a section of the document and return context around matches.
    
    Args:
        keyword: The keyword or phrase to search for
        start_line: Optional starting line to search from
    
    Returns:
        Lines containing the keyword with context, or a message if not found in that section.
    """
    try:
        with open('input.md', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Search in a 50-line window
        if start_line is not None and 0 <= start_line < total_lines:
            start = start_line
        else:
            start = random.randint(0, max(0, total_lines - 50))
        
        end = min(start + 50, total_lines)
        search_lines = lines[start:end]
        
        matches = []
        for i, line in enumerate(search_lines, start=start+1):
            if keyword.lower() in line.lower():
                # Include context (2 lines before and after)
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
            result += "\n".join(matches[:3])  # Limit to first 3 matches
            print(f"[TOOL] Search found {len(matches)} matches for '{keyword}' in lines {start+1}-{end}")
        else:
            result = f"No matches for '{keyword}' found in lines {start+1}-{end} of {total_lines}. Try searching another section."
            print(f"[TOOL] No matches for '{keyword}' in lines {start+1}-{end}")
        
        return result
        
    except FileNotFoundError:
        return "ERROR: input.md file not found!"
    except Exception as e:
        return f"ERROR: {str(e)}"

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration_count: int

# Initialize the model with localhost endpoint
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="gpt-oss",
    temperature=0.7,
    use_responses_api=True
)

# Bind tools to the model
tools = [read_document_chunk, search_document_section]
llm_with_tools = llm.bind_tools(tools)

# Single agent node that does the reasoning
def agent_node(state: AgentState) -> AgentState:
    """The main agent that reasons and decides which tools to call"""
    messages = state["messages"]
    iteration = state.get("iteration_count", 0) + 1
    
    print(f"\n{'='*80}")
    print(f"[AGENT] Iteration {iteration}")
    print(f"[AGENT] Message history size: {len(messages)} messages")
    
    # Calculate total tokens (rough estimate)
    total_chars = sum(len(str(msg.content)) if hasattr(msg, 'content') else 0 for msg in messages)
    print(f"[AGENT] Approximate message history size: {total_chars:,} characters")
    print(f"{'='*80}")
    
    # The agent receives ALL previous messages including tool results
    response = llm_with_tools.invoke(messages)
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[AGENT] Decision: Call {len(response.tool_calls)} tool(s)")
        for tc in response.tool_calls:
            print(f"  - {tc['name']} with args: {tc['args']}")
    else:
        print(f"[AGENT] Decision: Provide final answer")
        if hasattr(response, 'content'):
            preview = response.content[:150].replace('\n', ' ')
            print(f"[AGENT] Response preview: {preview}...")
    
    return {
        "messages": [response],
        "iteration_count": iteration
    }

# Single tool node
tool_node = ToolNode(tools)

# Router function
def should_continue(state: AgentState) -> str:
    """Determine if we should continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    iteration = state.get("iteration_count", 0)
    
    # Safety limit to prevent infinite loops (increased for document search)
    if iteration > 30:
        print(f"\n[ROUTER] Reached iteration limit ({iteration}), ending")
        return "end"
    
    # If there are tool calls, go back to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"[ROUTER] Tool calls detected -> routing to tool_node")
        return "continue"
    
    # Otherwise we're done
    print(f"[ROUTER] No tool calls -> ending")
    return "end"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# Always go back to agent after tools
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Test the agent
if __name__ == "__main__":
    # Query that will force multiple document reads
    query = """Please search through the input.md document and find the definition of "Power Systems". 
    
The document is large, so you'll need to read through it in chunks. Keep searching different sections 
until you find where "Power Systems" is defined or described. Once you find it, provide me with the 
complete definition and the line number where you found it.

Note: The document might not actually contain information about Power Systems, but I want you to 
thoroughly search through it to confirm whether it's there or not."""
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "iteration_count": 0
    }
    
    print("SEARCH QUERY:")
    print("=" * 80)
    print(query)
    print("=" * 80)
    print("\nSTARTING AGENT EXECUTION...\n")
    
    # Run the agent
    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total iterations: {final_state['iteration_count']}")
    print(f"Total messages in history: {len(final_state['messages'])}")
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    last_ai_message = [msg for msg in final_state['messages'] if isinstance(msg, AIMessage)][-1]
    print(last_ai_message.content)
    
    print("\n" + "=" * 80)
    print("MESSAGE HISTORY BREAKDOWN:")
    print("=" * 80)
    human_msgs = sum(1 for msg in final_state['messages'] if isinstance(msg, HumanMessage))
    ai_msgs = sum(1 for msg in final_state['messages'] if isinstance(msg, AIMessage))
    tool_msgs = sum(1 for msg in final_state['messages'] if isinstance(msg, ToolMessage))
    print(f"Human messages: {human_msgs}")
    print(f"AI messages: {ai_msgs}")
    print(f"Tool messages: {tool_msgs}")
    print(f"Total messages: {len(final_state['messages'])}")
    
    # Show tool call statistics
    tool_calls_made = sum(1 for msg in final_state['messages'] if isinstance(msg, ToolMessage))
    print(f"\nTotal tool calls made: {tool_calls_made}")