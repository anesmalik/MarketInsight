from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from tools import (
    get_stock_price, get_company_info, get_historical_data,
    get_balance_sheet, get_income_statement, get_cash_flow,
    get_financial_ratios, get_dividends, get_splits,
    get_major_holders, get_institutional_holders, get_insider_transactions,
    get_analyst_recommendations, get_price_targets, get_earnings, ticker_lookup
)

# 1. Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Collect all tools
tools = [
    get_stock_price, get_company_info, get_historical_data,
    get_balance_sheet, get_income_statement, get_cash_flow,
    get_financial_ratios, get_dividends, get_splits,
    get_major_holders, get_institutional_holders, get_insider_transactions,
    get_analyst_recommendations, get_price_targets, get_earnings, ticker_lookup
]

# 3. Create LLMs
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 4. System prompt with guardrail instruction
SYSTEM_PROMPT = """You are a financial analyst assistant with access to real-time market data.

IMPORTANT: You ONLY answer questions related to:
- Stock prices and market data
- Company financials (balance sheets, income statements, cash flow)
- Financial ratios and metrics
- Dividends, stock splits, and corporate actions
- Analyst recommendations and price targets
- Institutional and insider holdings

If a user asks about anything NOT related to finance, stocks, or markets, politely decline and explain that you only handle financial questions.

Examples of questions you should NOT answer:
- General knowledge questions
- Coding or technical help
- Personal advice unrelated to investing
- News or politics unrelated to markets

Always use your tools to fetch current data - never make up numbers."""

# 5. Input guardrail - checks if question is finance-related
GUARDRAIL_PROMPT = """You are a classifier. Determine if the user's message is related to finance, stocks, investing, or markets.

Respond with ONLY one word:
- "FINANCE" if the question is about stocks, markets, companies, investing, financial data, or economics
- "OTHER" if the question is about anything else

Examples:
- "What's Apple's stock price?" → FINANCE
- "Tell me a joke" → OTHER
- "How's the S&P 500 doing?" → FINANCE
- "What's the weather today?" → OTHER
- "Should I invest in tech stocks?" → FINANCE
- "Write me a poem" → OTHER
"""

def input_guardrail(message: str) -> bool:
    """Returns True if message is finance-related, False otherwise."""
    response = llm.invoke([
        SystemMessage(content=GUARDRAIL_PROMPT),
        HumanMessage(content=message)
    ])
    return "FINANCE" in response.content.upper()

# 6. The LLM node
def llm_node(state: State):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 7. The routing function
def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 8. Build the graph
graph_builder = StateGraph(State)

graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "llm")

agent = graph_builder.compile()