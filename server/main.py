from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage,AIMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Optional, Dict, Any, List, Union
from typing import Annotated, Sequence, List, Literal 
from langgraph.types import Command 
from pydantic import BaseModel, Field 
import pandas as pd


# Load environment variables
load_dotenv()
main = FastAPI(debug=True)
main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["Content-Type"], 
)

def serialise_ai_message_chunk(chunk): 
    if(isinstance(chunk, AIMessageChunk)):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )


# Initialize LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
search_tool = TavilySearchResults(max_results=1)
tools = [search_tool]
llm_with_tools = model.bind_tools(tools)

# Set up memory and tools
memory = MemorySaver()

# Create a mock JSON dataset for demonstration
MOCK_JSON = """
[
  {"date": "2023-01-01", "sales": 1200, "region": "North", "product": "Widget A"},
  {"date": "2023-01-01", "sales": 950, "region": "South", "product": "Widget A"},
  {"date": "2023-01-01", "sales": 1500, "region": "East", "product": "Widget B"},
  {"date": "2023-01-01", "sales": 1100, "region": "West", "product": "Widget B"},
  {"date": "2023-01-02", "sales": 1300, "region": "North", "product": "Widget A"},
  {"date": "2023-01-02", "sales": 1000, "region": "South", "product": "Widget A"},
  {"date": "2023-01-02", "sales": 1600, "region": "East", "product": "Widget B"},
  {"date": "2023-01-02", "sales": 1050, "region": "West", "product": "Widget B"},
  {"date": "2023-01-03", "sales": 1250, "region": "North", "product": "Widget A"},
  {"date": "2023-01-03", "sales": 980,  "region": "South", "product": "Widget A"},
  {"date": "2023-01-03", "sales": 1550, "region": "East", "product": "Widget B"},
  {"date": "2023-01-03", "sales": 1120, "region": "West", "product": "Widget B"},
  {"date": "2023-01-04", "sales": 1350, "region": "North", "product": "Widget A"},
  {"date": "2023-01-04", "sales": 1020, "region": "South", "product": "Widget A"},
  {"date": "2023-01-04", "sales": 1650, "region": "East", "product": "Widget B"},
  {"date": "2023-01-04", "sales": 1080, "region": "West", "product": "Widget B"}
]

"""

# Define the state type
class State(TypedDict):
    messages: Annotated[list, add_messages]
    request_type: str
    formula: Optional[str]
    dataframe: Optional[Dict[str, Any]]

# Supervisor model definition
class Supervisor(BaseModel):
    next: Literal["summary", "open_prompt", "model"] = Field(
        description="Determines which specialist to activate next in the workflow sequence: "
                    "'summary' When user ask question regarding summary of the table. "
                    "'open_prompt' When he asks question about data table. Eg) What is my highest or lowest "
                    "'model' When user ask about current happening or some thing that can be searched in google "
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )

def supervisor(state: State) -> Command:
    system_prompt = '''
Act like a supervisor agent in a multi-agent system. Your job is to decide which specialist to activate based on the user's question.

Return only one of the following strings:
- "summary" → when the user asks for a summary of a table.
- "open_prompt" → when the user asks specific questions about data, e.g., highest or lowest values.
- "model" → when the user asks about current events or things you'd find on Google.

Only return one of the four strings. Do not explain. Do not respond to the user.

Take a deep breath and work on this problem step-by-step.
'''
    
    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"] 

    response = model.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason
    
    return Command(    
        update={
            "request_type": goto,
            "messages": [
                HumanMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,  
    )

async def open_prompt(state: State) -> Dict:
    """
    Converts user query to a pandas DataFrame formula.
    """
    # Get the last human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    last_human_message = human_messages[-1].content
    data = json.loads(MOCK_JSON)
    df = pd.DataFrame(data)
    columns = df.columns.tolist()

# b = unique values per column, skipping 'sales'
    rowData = {col: df[col].unique().tolist() for col in df.columns if col != 'sales'}
    # Create a prompt for the LLM to generate a pandas formula
    prompt = f"""
    Convert the following user query into a pandas DataFrame formula.
    Use 'df' as the variable name for the DataFrame.
    
    User query: {last_human_message}
    "Columns:": {columns}
    "Unique values per column (excluding 'sales'):":{rowData}
    
    Example:
    If the query is "Find sales above 1200 in the East region", the formula would be:
    df[(df['sales'] > 1200) & (df['region'] == 'East')]
    
    Only return the pandas code, nothing else.
    """
    
    # Generate formula using LLM
    formula_result = await model.ainvoke([HumanMessage(content=prompt)])
    formula = formula_result.content.strip()
    
    # Clean up the formula (remove backticks if present)
    if formula.startswith("```python"):
        formula = formula.split("```python")[1]
    if formula.endswith("```"):
        formula = formula.split("```")[0]
    formula = formula.strip()
    
    # Update state with the formula
    return {
        "formula": formula,
        "messages": state["messages"]  # Keep existing messages
    }

async def open_prompt_validator(state: State) -> Union[Dict, str]:
    """
    Validates the formula by running it against the DataFrame.
    Uses the REPL tool to execute the formula and check the results.
    """
    # Get the formula from state
    formula = state.get("formula")
    if not formula:
        return {"messages": [AIMessage(content="No formula found to validate.")]}
    
    

    # This would normally use the actual REPL tool, but here we'll simulate it
    try:
        # In a real implementation, you would use something like:
        # repl_result = await repl_tool.ainvoke({"code": code})
        
        # For demonstration, let's simulate REPL execution
        # Parse mock JSON data
        data = json.loads(MOCK_JSON)
        df = pd.DataFrame(data)
        
        # Safely evaluate the formula (note: this is simplified for demo)
        # In production, use a proper REPL or sanitized eval
        local_vars = {"df": df}
        try:
            # Very unsafe in production, use proper repl tool instead
            result_df = eval(formula, {"__builtins__": {}}, local_vars)
            if not isinstance(result_df, pd.DataFrame):
                if isinstance(result_df, pd.Series):
                    result_df = result_df.to_frame()
                else:
                    result_df = pd.DataFrame()
            
            row_count = len(result_df)
            
            if row_count > 0:
                result_preview = result_df.head(10).to_dict(orient='records')
                result_message = f"Preview: {json.dumps(result_preview, indent=2)}"
                prompt = f"""
                    Act as a data expert analyst ,
                    Please summarize the following data :
                    {result_message}
                    Provide key insights about the data.Make sure its consice.
                    """
        
                summary_result = await model.ainvoke([HumanMessage(content=prompt)])
                return {
                    "messages": [AIMessage(content=summary_result.content)],
                    "request_type": "open_prompt"

                }
          
            else:
                # No rows found, go back to formula node
                return {
                    "messages": [AIMessage(content="Please try again!!!!")],
                    "request_type": "open_prompt"

                }
                
        except Exception as e:
            # Formula error, go back to formula node
            error_message = f"Error in formula: {str(e)}. Let me try a different approach."
            return {
                "messages": [AIMessage(content=error_message)],
                "formula": None  # Clear the invalid formula
            }
            
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Validation error: {str(e)}")],
            "formula": None  # Clear the formula on error
        }
async def summary(state: State) -> Dict:
    """
    Converts mock JSON to DataFrame, calculates summary statistics,
    and updates the state with this information.
    """
    try:
        data = json.loads(MOCK_JSON)
        df = pd.DataFrame(data)
        
        # Calculate summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        summary_data = {}
        
        for col in numeric_cols:
            summary_data[col] = {
                "min": df[col].min(),
                "max": df[col].max(),
                "avg": df[col].mean()
            }
        
        # Prepare data for LLM summarization
        prompt = f"""
        Please summarize the following data statistics:
        {summary_data}
        Provide key insights about the data . Make sure its consice.
        """
        
        summary_result = await model.ainvoke([HumanMessage(content=prompt)])
        
        # Return updated state with summary data
        return {
            "messages": [AIMessage(content=summary_result.content)],
            "request_type": "summary_chat"
        }
    
    except Exception as e:
        error_msg = f"Error in summary node: {str(e)}"
        return {"messages": [AIMessage(content=error_msg)]}


async def model_node(state: State) -> Dict:
    """LLM node that processes messages."""
    result = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": [result],
        "request_type": "model_response"
    }

async def tools_router(state: State) -> str:
    """Routes to tool_node if the last message has tool calls, otherwise ends."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else: 
        return END

async def tool_node(state: State) -> Dict:
    """Custom tool node that handles tool calls from the LLM."""
    # Get the tool calls from the last message
    tool_calls = state["messages"][-1].tool_calls
    
    # Initialize list to store tool messages
    tool_messages = []
    
    # Process each tool call
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Handle the search tool
        if tool_name == "tavily_search_results_json":
            # Execute the search tool with the provided arguments
            search_results = await search_tool.ainvoke(tool_args)
            
            # Create a ToolMessage for this result
            tool_message = ToolMessage(
                content=str(search_results),
                tool_call_id=tool_id,
                name=tool_name
            )
            
            tool_messages.append(tool_message)
    
    # Add the tool messages to the state
    return {"messages": tool_messages}

def build_graph():
    # Initialize the graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("summary", summary)
    graph_builder.add_node("model", model_node)
    graph_builder.add_node("open_prompt",open_prompt)
    graph_builder.add_node("open_prompt_validator",open_prompt_validator)
    graph_builder.add_node("tool_node", tool_node)

    # Set entry point
    graph_builder.set_entry_point("supervisor")
    
    # Add edges
    graph_builder.add_edge("summary", END)
    
    # Conditional edges for tool handling
    graph_builder.add_conditional_edges("model", tools_router)
    graph_builder.add_edge("tool_node", "model")
    graph_builder.add_edge("open_prompt","open_prompt_validator")
    graph_builder.add_edge("open_prompt_validator", END)

    
    # Compile the graph
    return graph_builder.compile(checkpointer=memory)

# Main execution function
graph = build_graph()



async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    
    if is_new_conversation:
        # Generate new checkpoint ID for first message in conversation
        new_checkpoint_id = str(uuid4())

        config = {
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }
        
        # Initialize with first message
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
        
        # First send the checkpoint ID
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {
            "configurable": {
                "thread_id": checkpoint_id
            }
        }
        # Continue existing conversation
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    async for event in events:
        event_type = event["event"]
        
        if event_type == "on_chat_model_stream":
           if (
                isinstance(event, dict)
                and 'metadata' in event
                and isinstance(event['metadata'], dict)
                and event['metadata'].get('langgraph_node') not in ('supervisor', 'open_prompt')): 
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            # Escape single quotes and newlines for safe JSON parsing
            safe_content = chunk_content.replace("'", "\\'").replace("\n", "\\n")
            
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"
            
        elif event_type == "on_chat_model_end":
            # Check if there are tool calls for search
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "tavily_search_results_json"]
            
            if search_calls:
                # Signal that a search is starting
                search_query = search_calls[0]["args"].get("query", "")
                # Escape quotes and special characters
                safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"
                
        elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
            # Search completed - send results or error
            output = event["data"]["output"]
            
            # Check if output is a list 
            if isinstance(output, list):
                # Extract URLs from list of search results
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])
                
                # Convert URLs to JSON and yield them
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"
    
    # Send an end event
    yield f"data: {{\"type\": \"end\"}}\n\n"


@main.get("/stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream"
    )