# MarketInsight — Full-Stack AI Agent Training Curriculum

> **Purpose**: A step-by-step training guide for building an AI-powered stock market analysis platform from scratch. Each module builds on the previous one, producing a working component that feeds into the final system.
>
> **Final Product**: A conversational AI agent that answers stock market questions by orchestrating 16 specialized financial tools, served via a FastAPI backend with streaming responses and a React frontend.
>
> **Tech Stack**: Python · FastAPI · LangChain · LangGraph · OpenAI GPT · YFinance · Langfuse · React · TypeScript

---

## Architecture Overview

Before diving into modules, here's how all the pieces connect:

```
┌─────────────────────────────────────────────────────────┐
│                     FRONTEND (React)                     │
│  Chat UI → Streaming Display → Markdown Rendering        │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/SSE (Streaming)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  BACKEND (FastAPI)                        │
│  /chat endpoint → Session Management → CORS              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              AGENT ORCHESTRATION (LangGraph)              │
│                                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐    │
│  │  START    │───▶│ LLM Node │───▶│ Conditional Edge │    │
│  └──────────┘    │ (Reason) │    │ (Tool call? or   │    │
│                  └──────────┘    │  final answer?)  │    │
│                       ▲          └────────┬─────────┘    │
│                       │                   │              │
│                       │          ┌────────▼─────────┐    │
│                       │          │    Tool Node      │    │
│                       └──────────│   (Execute)       │    │
│                                  └──────────────────┘    │
│                                          │               │
│                                  ┌───────▼───────┐       │
│                                  │     END        │       │
│                                  └───────────────┘       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               TOOLS (LangChain @tool)                    │
│  get_stock_price · get_historical_data · balance_sheet   │
│  income_statement · cash_flow · company_info · ratios    │
│  dividends · splits · holders · insider_transactions     │
│  analyst_recommendations · ticker_lookup · ...           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              DATA SOURCE (Yahoo Finance API)             │
│  yfinance library → Real-time & historical market data   │
└─────────────────────────────────────────────────────────┘

           ┌──────────────────────────────┐
           │   OBSERVABILITY (Langfuse)    │
           │  Traces every LLM call,       │
           │  tool invocation, latency,    │
           │  token usage, and cost        │
           └──────────────────────────────┘
```

**Data flow for a single user query** ("What's Apple's P/E ratio compared to its 5-year average?"):

1. User types in React frontend → sends HTTP POST to `/chat`
2. FastAPI receives request → passes to LangGraph agent
3. LangGraph's LLM node reasons: "I need the current P/E ratio and historical data"
4. Conditional edge routes to Tool Node → calls `get_financial_ratios("AAPL")` and `get_historical_data("AAPL")`
5. Tool results return to LLM node → LLM synthesizes a response
6. Conditional edge routes to END → response streams back through FastAPI → React renders it token by token
7. Langfuse captures the entire trace: LLM calls, tool calls, latencies, token counts

---

## Module 1: Foundations — YFinance Data Extraction

### What You're Building
A Python module that fetches all types of financial data from Yahoo Finance. This becomes the data layer that every tool in the agent wraps.

### Why This Module Exists
The MarketInsight agent has 16 tools. Every single one is a thin wrapper around a YFinance call. If you don't understand the raw data — its shape, its quirks, its failure modes — your tools will be unreliable, and an unreliable tool makes an unreliable agent. You need to know what data is available, what format it comes in, and where it breaks.

### Core Concepts

#### 1.1 What is YFinance?
`yfinance` is a Python library that scrapes data from Yahoo Finance. It's not an official API — Yahoo doesn't provide one anymore — which means:
- It can break when Yahoo changes their website structure
- Rate limiting is unofficial and unpredictable
- Some data fields may be None/NaN unexpectedly
- It's free, which is why every financial hobby project uses it

This matters for your agent design: every tool needs to handle missing data gracefully, because YFinance will return incomplete data more often than you'd expect.

#### 1.2 The Ticker Object
Everything in YFinance starts with a `Ticker` object. Think of it as a connection to all data about a single stock:

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")  # Creates a Ticker object for Apple
```

From this single object, you can access:

| Property/Method | Returns | Data Type |
|----------------|---------|-----------|
| `ticker.info` | Company profile, current price, ratios | Dict |
| `ticker.history(period="1mo")` | Historical OHLCV data | DataFrame |
| `ticker.balance_sheet` | Balance sheet | DataFrame |
| `ticker.income_stmt` | Income statement | DataFrame |
| `ticker.cashflow` | Cash flow statement | DataFrame |
| `ticker.dividends` | Dividend history | Series |
| `ticker.splits` | Stock split history | Series |
| `ticker.recommendations` | Analyst recommendations | DataFrame |
| `ticker.institutional_holders` | Top institutional holders | DataFrame |
| `ticker.major_holders` | Major holders breakdown | DataFrame |
| `ticker.insider_transactions` | Insider buy/sell activity | DataFrame |

#### 1.3 Data Categories Deep Dive

**Current Price & Quote Data**
The `info` dict is massive (100+ keys). The ones that matter for a financial analysis agent:

```python
info = ticker.info

# Price data
info['currentPrice']        # Current trading price
info['previousClose']       # Yesterday's close
info['open']                # Today's open
info['dayHigh']             # Today's high
info['dayLow']              # Today's low
info['volume']              # Current volume
info['averageVolume']       # Average volume

# Valuation ratios
info['trailingPE']          # Price/Earnings (trailing 12 months)
info['forwardPE']           # Price/Earnings (forward estimate)
info['priceToBook']         # Price/Book ratio
info['priceToSalesTrailing12Months']

# Company profile
info['longName']            # Full company name
info['sector']              # e.g., "Technology"
info['industry']            # e.g., "Consumer Electronics"
info['longBusinessSummary'] # Company description
info['website']             # Company URL
info['country']             # Country of HQ
info['fullTimeEmployees']   # Employee count

# Financial health
info['debtToEquity']        # Debt/Equity ratio
info['returnOnEquity']      # ROE
info['profitMargins']       # Profit margin
info['operatingMargins']    # Operating margin
info['freeCashflow']        # Free cash flow
```

**Key gotcha**: Not every key exists for every ticker. A mutual fund won't have `trailingPE`. An ETF won't have `fullTimeEmployees`. Your tools must handle `KeyError` and `None` values.

**Historical Data**
This is the most commonly requested data. The `history()` method returns OHLCV (Open, High, Low, Close, Volume) data:

```python
# Different time periods
hist = ticker.history(period="1d")    # Today
hist = ticker.history(period="5d")    # 5 days
hist = ticker.history(period="1mo")   # 1 month
hist = ticker.history(period="1y")    # 1 year
hist = ticker.history(period="max")   # All available

# Custom date range
hist = ticker.history(start="2023-01-01", end="2024-01-01")

# Different intervals
hist = ticker.history(period="1mo", interval="1d")   # Daily
hist = ticker.history(period="5d", interval="1h")     # Hourly
hist = ticker.history(period="1d", interval="5m")     # 5-minute
```

The returned DataFrame has columns: `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits`. The index is a DatetimeIndex.

**Key gotcha**: The `history()` method returns adjusted prices by default (adjusted for splits and dividends). If you need raw prices, pass `auto_adjust=False`.

**Financial Statements**
Three core statements, each returning a DataFrame where columns are dates (quarters or annual) and rows are line items:

```python
# Annual statements
bs = ticker.balance_sheet       # Balance Sheet
is_ = ticker.income_stmt        # Income Statement
cf = ticker.cashflow            # Cash Flow Statement

# Quarterly statements
bs_q = ticker.quarterly_balance_sheet
is_q = ticker.quarterly_income_stmt
cf_q = ticker.quarterly_cashflow
```

The column headers are Timestamp objects (e.g., `Timestamp('2024-09-28')`), and row indices are strings like `'Total Revenue'`, `'Net Income'`, `'Total Assets'`, etc.

**Key gotcha**: The row labels are not standardized across companies. Some companies report `'Total Revenue'`, others `'Revenue'`. Your tools should handle both or use fuzzy matching.

#### 1.4 Error Handling Patterns
YFinance fails in specific, predictable ways:

```python
# Invalid ticker — doesn't throw an error, returns empty data
bad = yf.Ticker("NOTREAL")
bad.info  # Returns minimal dict or raises exception

# Delisted stock — may return partial historical data but empty financials
delisted = yf.Ticker("LEHM")  # Lehman Brothers

# Rate limiting — too many requests in quick succession
# No error thrown, but data may be stale or incomplete
```

**Design principle for your tools**: Always validate that the data you got back is actually useful before returning it to the LLM. An empty DataFrame is worse than a clear error message, because the LLM will try to analyze empty data and hallucinate.

#### 1.5 Data Serialization for LLM Consumption
The LLM can't read a pandas DataFrame directly. You need to convert data into a format the LLM can reason about. This is a critical design decision that affects the quality of the agent's responses:

```python
# Option A: Convert to string (simple but verbose)
df.to_string()

# Option B: Convert to dict (structured but can be huge)
df.to_dict()

# Option C: Convert to markdown table (LLM-friendly)
df.to_markdown()

# Option D: Selective summary (best for large datasets)
f"Latest close: {df['Close'].iloc[-1]}, 30-day avg: {df['Close'].mean():.2f}"
```

The choice depends on context. For small datasets (a few rows), markdown tables work well. For large historical datasets (years of daily data), you need to summarize — sending 1,000 rows to the LLM wastes tokens and confuses the model.

### Exercises for This Module
1. Write a function that fetches complete company info for a ticker and handles invalid tickers gracefully
2. Write a function that fetches historical data and returns it as a formatted string suitable for LLM consumption (think about what the LLM actually needs)
3. Write a function that fetches all three financial statements and identifies the most recent quarter's data
4. Test your functions with edge cases: ETFs, mutual funds, delisted stocks, crypto tickers (BTC-USD)

### Key Decisions to Discuss Before Coding
- How should we format DataFrame output for the LLM? (Token efficiency vs. completeness)
- Should we cache YFinance responses? (Rate limiting vs. data freshness)
- How do we handle the ticker lookup problem? (User says "Apple", tool needs "AAPL")

---

## Module 2: LangChain Tools — Wrapping YFinance as Callable Tools

### What You're Building
A collection of 16 LangChain tool functions that the AI agent can call. Each tool wraps a specific YFinance data retrieval operation with proper descriptions, input validation, and error handling.

### Why This Module Exists
An LLM without tools is just a chatbot that can only use its training data. Tools give it the ability to take actions — in this case, fetching live financial data. But the tool design is critical: the LLM chooses which tool to call based entirely on the tool's name and description. A bad description = the LLM picks the wrong tool or passes wrong parameters.

### Core Concepts

#### 2.1 What is a LangChain Tool?
A LangChain tool is a Python function that:
1. Has a clear **name** the LLM can reference
2. Has a **description** the LLM reads to understand when to use it
3. Has **typed parameters** the LLM knows how to fill in
4. Returns a **string** the LLM can reason about

```python
from langchain_core.tools import tool

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL' for Apple, 'GOOGL' for Google)
    
    Returns:
        Current price information including open, close, high, low, and volume.
    """
    # Implementation here
    pass
```

The `@tool` decorator does several things:
- Extracts the function name as the tool name
- Extracts the docstring as the tool description
- Infers the input schema from type hints
- Wraps the function so LangChain can manage it

#### 2.2 The 16 Tools of MarketInsight
Based on the repo's API capabilities, here are the tools you need to build:

| # | Tool Name | Purpose | Input |
|---|-----------|---------|-------|
| 1 | `get_stock_price` | Current price data | ticker symbol |
| 2 | `get_historical_data` | Historical OHLCV | ticker, period, interval |
| 3 | `get_balance_sheet` | Balance sheet | ticker, quarterly flag |
| 4 | `get_income_statement` | Income statement | ticker, quarterly flag |
| 5 | `get_cash_flow` | Cash flow statement | ticker, quarterly flag |
| 6 | `get_company_info` | Company profile | ticker |
| 7 | `get_financial_ratios` | Valuation ratios | ticker |
| 8 | `get_dividend_history` | Dividend payments | ticker |
| 9 | `get_stock_splits` | Split history | ticker |
| 10 | `get_major_holders` | Major holder breakdown | ticker |
| 11 | `get_institutional_holders` | Top institutions | ticker |
| 12 | `get_insider_transactions` | Insider buy/sell | ticker |
| 13 | `get_analyst_recommendations` | Buy/sell/hold ratings | ticker |
| 14 | `get_ticker_lookup` | Symbol from name | company name |
| 15 | `get_earnings_history` | Quarterly EPS | ticker |
| 16 | `get_news` | Recent news articles | ticker |

#### 2.3 Writing Effective Tool Descriptions
This is arguably the most important part of tool design. The LLM reads the description to decide:
- **When** to use this tool vs. another
- **What** parameters to pass
- **What** to expect back

Bad description:
```python
@tool
def get_data(ticker: str) -> str:
    """Gets data for a stock."""
```

Good description:
```python
@tool
def get_financial_ratios(ticker: str) -> str:
    """Get key financial valuation ratios for a stock.
    
    Use this tool when the user asks about P/E ratio, price-to-book,
    debt-to-equity, profit margins, return on equity, or other 
    financial metrics. Do NOT use this for historical price data 
    (use get_historical_data instead) or for company profile 
    information (use get_company_info instead).
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    
    Returns:
        Key ratios including P/E, P/B, D/E, ROE, margins, and more.
    """
```

Notice the description tells the LLM:
- What this tool is **for** (financial ratios)
- What questions it **answers** (P/E, price-to-book, etc.)
- What it is **NOT for** (prevents confusion with similar tools)
- What **input format** it expects (ticker symbols with examples)

#### 2.4 Input Validation
The LLM can and will pass garbage inputs. Your tools need to survive this:

```python
@tool
def get_stock_price(ticker: str) -> str:
    """..."""
    try:
        stock = yf.Ticker(ticker.upper().strip())
        info = stock.info
        
        # Validate we got real data back
        if not info or 'currentPrice' not in info:
            return f"Could not find stock data for ticker '{ticker}'. Please verify the ticker symbol is correct."
        
        # Format response for LLM
        return f"""
Stock: {info.get('longName', ticker)} ({ticker.upper()})
Current Price: ${info['currentPrice']:.2f}
Previous Close: ${info.get('previousClose', 'N/A')}
Day Range: ${info.get('dayLow', 'N/A')} - ${info.get('dayHigh', 'N/A')}
Volume: {info.get('volume', 'N/A'):,}
"""
    except Exception as e:
        return f"Error fetching stock price for '{ticker}': {str(e)}"
```

Key patterns:
- **Always uppercase the ticker** (LLM might pass "aapl")
- **Validate the response** before formatting
- **Return error messages as strings** (not exceptions) — the LLM can read error messages and try a different approach
- **Use `.get()` with defaults** for optional fields

#### 2.5 Return Format Design
What you return from a tool directly affects the quality of the agent's response. The LLM will see this text and use it to formulate its answer.

**Principles**:
- Return **structured text**, not raw data dumps
- Include **labels** for every value (the LLM needs context)
- **Summarize large datasets** — don't return 500 rows
- Include the **ticker symbol** in the response (the LLM might be juggling multiple stocks)
- Use **units** — "$45.23" not "45.23"

#### 2.6 The Ticker Lookup Problem
Users say "Apple" or "Microsoft", but YFinance needs "AAPL" or "MSFT". The `get_ticker_lookup` tool solves this, but there are design choices:

Option A: Use `yfinance.search()` (unreliable)
Option B: Use a hardcoded mapping of popular stocks
Option C: Let the LLM figure it out (it knows most major tickers)
Option D: Use an external API for ticker search

The MarketInsight project likely uses a combination — the LLM's knowledge for major stocks, plus a lookup tool for ambiguous cases.

#### 2.7 Tool Testing
Every tool should work independently before plugging into the agent. Test pattern:

```python
# Direct invocation (bypasses LangChain wrapper)
result = get_stock_price.invoke({"ticker": "AAPL"})
print(result)

# Test edge cases
result = get_stock_price.invoke({"ticker": "INVALID"})
print(result)  # Should return helpful error, not crash
```

### Exercises for This Module
1. Implement the first 5 tools (price, historical, balance sheet, income statement, cash flow)
2. Write descriptions that clearly differentiate between get_financial_ratios, get_company_info, and get_stock_price
3. Test each tool with valid tickers, invalid tickers, and edge cases
4. Experiment with different return formats and observe how they affect token count

### Key Decisions to Discuss Before Coding
- Should tools accept additional parameters (like date range for historical data) or keep it simple?
- How much data should each tool return? (Token budget per tool call)
- Should the ticker lookup be a separate tool or built into every tool?
- How should we handle tools that return DataFrames with many columns?

---

## Module 3: LangChain + OpenAI — Basic Chat with Tool Calling

### What You're Building
A working conversational system where you chat with an OpenAI model and it automatically decides which YFinance tools to call, executes them, reads the results, and synthesizes a response.

### Why This Module Exists
Before jumping to LangGraph orchestration, you need to understand the raw mechanics of tool calling. What actually happens when an LLM "calls a tool"? How does the message flow work? What does the LLM actually see? This module removes the LangGraph abstraction so you understand the fundamentals.

### Core Concepts

#### 3.1 The Tool Calling Flow
When you bind tools to an LLM, here's what happens on each turn:

```
User Message
    │
    ▼
┌─────────────────────────┐
│  LLM receives:          │
│  - System prompt         │
│  - Conversation history  │
│  - Tool definitions      │
│  - User's question       │
└───────────┬─────────────┘
            │
            ▼
    LLM decides: "Do I need to call a tool?"
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
  YES: Tool     NO: Direct
  Call          Response
     │             │
     ▼             ▼
┌──────────┐   ┌──────────┐
│ Returns   │   │ Returns   │
│ AIMessage │   │ AIMessage │
│ with      │   │ with      │
│ tool_calls│   │ content   │
│ field     │   │ field     │
└─────┬────┘   └──────────┘
      │
      ▼
┌──────────────────┐
│ Execute tool(s)   │
│ Get result(s)     │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ Send tool results │
│ back to LLM       │
└─────┬────────────┘
      │
      ▼
┌──────────────────┐
│ LLM synthesizes   │
│ final response     │
└──────────────────┘
```

**Critical insight**: The LLM doesn't "execute" anything. It returns a structured message saying "I want to call function X with arguments Y." Your code is responsible for actually calling the function and feeding the result back.

#### 3.2 Setting Up the LLM with Tools

```python
from langchain_openai import ChatOpenAI

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o-mini",    # or "gpt-4o" for better reasoning
    temperature=0,            # 0 for consistent financial data responses
    api_key="your-key"
)

# Bind tools to the model
tools = [get_stock_price, get_historical_data, get_financial_ratios, ...]
llm_with_tools = llm.bind_tools(tools)
```

`bind_tools()` doesn't change the model — it attaches the tool schemas to every API call so the model knows what tools are available.

#### 3.3 Message Types
LangChain uses typed messages to represent the conversation:

```python
from langchain_core.messages import (
    SystemMessage,     # Instructions to the LLM
    HumanMessage,      # User input
    AIMessage,         # LLM response (may contain tool_calls)
    ToolMessage,       # Result of a tool execution
)
```

A complete conversation with tool calling looks like this in the messages list:

```python
messages = [
    SystemMessage(content="You are a financial analyst..."),
    HumanMessage(content="What's Apple's P/E ratio?"),
    AIMessage(content="", tool_calls=[{
        "name": "get_financial_ratios",
        "args": {"ticker": "AAPL"},
        "id": "call_abc123"
    }]),
    ToolMessage(content="P/E Ratio: 28.5, ...", tool_call_id="call_abc123"),
    AIMessage(content="Apple's current P/E ratio is 28.5, which means...")
]
```

Notice:
- The AIMessage with tool_calls has **empty content** — the model is saying "I need to call a tool before I can respond"
- The ToolMessage **must reference the tool_call_id** — this is how the LLM knows which tool result corresponds to which call
- The final AIMessage has **content** — this is the actual response to the user

#### 3.4 The System Prompt
The system prompt shapes the agent's personality, behavior, and domain expertise. For a financial analysis agent:

```python
SYSTEM_PROMPT = """You are MarketInsight, an AI-powered stock market analyst. 
You help users understand stock performance, financial health, and market trends.

Your capabilities:
- Fetch real-time stock prices and historical data
- Analyze financial statements (balance sheet, income statement, cash flow)
- Evaluate financial ratios and valuation metrics
- Track dividends, splits, and ownership data
- Provide analyst recommendations and insider transaction data

Guidelines:
- Always use tools to get current data. Never make up financial numbers.
- When comparing stocks, fetch data for all stocks before making comparisons.
- Present financial data clearly with proper formatting.
- If a tool returns an error, explain what happened and suggest alternatives.
- For ambiguous company names, use the ticker lookup tool first.
- Always mention that this is for informational purposes, not financial advice.
"""
```

**Design considerations**:
- Tell the LLM what tools it has (reinforces the tool descriptions)
- Set behavioral rules (never fabricate numbers, always disclaim)
- Guide multi-step reasoning (compare → fetch all → then analyze)

#### 3.5 The Manual Tool Execution Loop
Before LangGraph automates this, here's the raw loop:

```python
def chat(user_input: str, messages: list) -> str:
    messages.append(HumanMessage(content=user_input))
    
    # First LLM call — may return tool calls or direct response
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    # Loop while the LLM wants to call tools
    while response.tool_calls:
        # Execute each tool call
        for tool_call in response.tool_calls:
            # Find the right tool
            tool = tool_map[tool_call["name"]]
            # Execute it
            result = tool.invoke(tool_call["args"])
            # Add result to messages
            messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            ))
        
        # Send tool results back to LLM for next decision
        response = llm_with_tools.invoke(messages)
        messages.append(response)
    
    return response.content
```

**Why a loop?** The LLM might need multiple rounds of tool calls. For example:
- "Compare Apple and Microsoft" → calls get_stock_price("AAPL"), get_stock_price("MSFT")
- "Show me Apple's earnings trend and explain using their revenue data" → calls get_earnings_history("AAPL"), then maybe get_income_statement("AAPL") for more detail

#### 3.6 Parallel Tool Calls
Modern OpenAI models can request **multiple tool calls in a single response**. The `tool_calls` list may contain 2-3 calls:

```python
# LLM response for "Compare Apple and Microsoft stock prices"
AIMessage(tool_calls=[
    {"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "call_1"},
    {"name": "get_stock_price", "args": {"ticker": "MSFT"}, "id": "call_2"},
])
```

Your execution loop must handle this — execute all tool calls, then send all results back together.

#### 3.7 Temperature and Model Selection
For a financial analysis agent:
- **Temperature 0**: Consistent, deterministic responses. When someone asks for Apple's P/E ratio, they want the same answer every time.
- **gpt-4o-mini**: Good balance of cost and capability for tool routing. It's excellent at knowing *which* tool to call.
- **gpt-4o**: Better for complex multi-step analysis and synthesizing large amounts of financial data into coherent narratives.

The MarketInsight project lets you configure this — it's a trade-off between cost and quality.

### Exercises for This Module
1. Set up the basic LLM with 3-4 tools bound to it
2. Implement the manual tool execution loop
3. Test with progressively complex queries:
   - "What's Apple's stock price?" (single tool call)
   - "Compare Apple and Google P/E ratios" (parallel tool calls)
   - "Is Apple overvalued based on its financial ratios and recent earnings?" (multi-step reasoning)
4. Observe how the LLM decides which tool to call and examine the message flow

### Key Decisions to Discuss Before Coding
- Should we use gpt-4o-mini or gpt-4o? (Cost vs. quality trade-off for the default model)
- How should the system prompt guide tool usage vs. letting the model figure it out?
- Should we add a tool_map dict or use LangChain's built-in tool routing?
- How do we handle the case where the LLM calls a tool that doesn't exist?

---

## Module 4: LangGraph — Agent Orchestration

### What You're Building
A LangGraph `StateGraph` that replaces the manual tool-calling loop from Module 3 with a proper graph-based agent. This is the core of the MarketInsight `components/` directory.

### Why This Module Exists
The manual loop from Module 3 works, but it's fragile. It doesn't handle errors well, it can't be easily debugged, it has no concept of state management, and it can't be extended with features like human-in-the-loop or checkpointing. LangGraph gives you a structured, debuggable, extensible agent execution framework.

### Core Concepts

#### 4.1 Why LangGraph Over a Simple Loop?
The manual loop has problems:
- **No state management**: If the loop crashes, you lose everything
- **No visibility**: You can't see what the agent is "thinking" at each step
- **No extensibility**: Adding features like timeout, retry, or branching means rewriting the loop
- **No production readiness**: Can't checkpoint, resume, or stream intermediate steps

LangGraph solves these with a **graph-based execution model**: nodes do work, edges route between them, and state flows through the entire graph.

#### 4.2 The State
State is the data that flows through the graph. For a chat agent, it's the conversation messages:

```python
from langgraph.graph import MessagesState

# MessagesState is a TypedDict with a single key:
# { "messages": list[BaseMessage] }
# It automatically handles message appending (not replacing)
```

`MessagesState` is a convenience class. Under the hood, it's:

```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

The `add_messages` annotation is crucial — it means when a node returns `{"messages": [new_message]}`, the new message is **appended** to the list, not replacing it. This is how conversation history accumulates.

You can extend the state with additional fields if needed:

```python
class MarketInsightState(TypedDict):
    messages: Annotated[list, add_messages]
    current_ticker: str  # Track which stock we're analyzing
    tool_call_count: int  # Limit tool calls to prevent infinite loops
```

#### 4.3 Nodes
Nodes are functions that take the current state and return updates to it:

```python
def llm_node(state: MessagesState):
    """The reasoning node — calls the LLM to decide what to do next."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

```python
from langgraph.prebuilt import ToolNode

# The tool execution node — runs whatever tools the LLM requested
tool_node = ToolNode(tools)
```

`ToolNode` is a prebuilt LangGraph node that:
1. Reads the last AIMessage's `tool_calls`
2. Executes each tool
3. Returns the results as ToolMessages
4. Handles errors (returns error messages instead of crashing)

#### 4.4 Edges and Routing
Edges define how the graph flows. The critical piece is the **conditional edge** that decides: did the LLM want to call tools, or is it done?

```python
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState):
    """Route based on whether the LLM wants to call tools."""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        return "tools"   # Route to tool node
    return END           # Route to end (response is ready)
```

#### 4.5 Building the Graph
Putting it all together:

```python
# 1. Create the graph with state schema
graph = StateGraph(MessagesState)

# 2. Add nodes
graph.add_node("agent", llm_node)       # The LLM reasoning node
graph.add_node("tools", tool_node)       # The tool execution node

# 3. Add edges
graph.add_edge(START, "agent")           # Start → Agent
graph.add_conditional_edges(             # Agent → Tools or END
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent")         # Tools → back to Agent

# 4. Compile
app = graph.compile()
```

The compiled graph looks like:

```
START → agent → [conditional] → tools → agent → [conditional] → END
                     │                                  │
                     └──────── END ◄────────────────────┘
```

This is the **ReAct loop**: Reason (agent node) → Act (tool node) → Observe (results back to agent) → Reason again → until done.

#### 4.6 Invoking the Graph

```python
# Simple invocation
result = app.invoke({
    "messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="What's Apple's stock price?")
    ]
})

# The result contains the full message history
final_response = result["messages"][-1].content
```

#### 4.7 Streaming
For production use, you want to stream the agent's work as it happens:

```python
# Stream events as the graph executes
for event in app.stream({
    "messages": [SystemMessage(content=SYSTEM_PROMPT), 
                 HumanMessage(content="Analyze Apple stock")]
}):
    for node_name, node_output in event.items():
        print(f"Node '{node_name}':")
        print(node_output)
```

This is how the FastAPI backend will work — streaming events to the frontend as the agent reasons and calls tools.

#### 4.8 Graph Configuration
LangGraph supports configuration that gets passed through the graph:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    configurable={
        "model": "gpt-4o",
        "temperature": 0,
    },
    callbacks=[langfuse_handler],  # Observability (Module 6)
)

result = app.invoke({"messages": [...]}, config=config)
```

#### 4.9 Error Handling in the Graph
What happens when a tool fails? `ToolNode` handles this by default — it returns an error message as a ToolMessage, and the LLM gets to decide what to do next (retry, try a different approach, or tell the user).

For additional safety, you can add:
- **Max iterations**: Limit how many times the agent can loop (prevent infinite tool calls)
- **Timeouts**: Limit total execution time
- **Fallback responses**: If the agent gets stuck, return a helpful message

```python
# One approach: Track iterations in state
class MarketInsightState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int

def should_continue(state):
    if state.get("iteration_count", 0) > 10:
        return END  # Safety valve
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

#### 4.10 The Prebuilt Alternative
LangGraph also offers a prebuilt `create_react_agent` that handles all of the above automatically:

```python
from langgraph.prebuilt import create_react_agent

app = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
```

This is simpler but less customizable. The MarketInsight project likely uses the manual graph construction (as shown in 4.5) for more control over the flow.

### Exercises for This Module
1. Build the StateGraph with the LLM node and ToolNode
2. Add conditional routing between tool calls and final response
3. Test with the same queries from Module 3 and compare behavior
4. Add a max iteration limit to prevent runaway tool calls
5. Use `app.stream()` to observe the agent's step-by-step execution
6. Visualize the graph using `app.get_graph().draw_mermaid()`

### Key Decisions to Discuss Before Coding
- Use `MessagesState` as-is or extend with custom fields?
- Use `ToolNode` prebuilt or write a custom tool execution node?
- Use `create_react_agent` shortcut or build the graph manually?
- What should the max iteration limit be?
- How should we handle tool timeouts?

---

## Module 5: FastAPI Backend — Exposing the Agent as an API

### What You're Building
The `main.py` FastAPI server that wraps the LangGraph agent in HTTP endpoints, with streaming response support so the frontend can display tokens as they arrive.

### Why This Module Exists
The LangGraph agent from Module 4 runs in a Python process. To make it accessible to a web frontend (or any client), it needs to be an HTTP API. FastAPI is the choice here because it natively supports async operations and streaming responses — both essential for a chat agent that needs to stream tokens as the LLM generates them.

### Core Concepts

#### 5.1 FastAPI Basics for AI Applications
FastAPI is an async Python web framework. For an AI agent, the key features are:
- **Async support**: LLM calls take seconds. Async prevents blocking other requests.
- **StreamingResponse**: Send tokens to the client as they're generated, not all at once.
- **Pydantic models**: Type-safe request/response schemas.
- **CORS middleware**: Allow the React frontend to talk to the API.

#### 5.2 Application Structure

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MarketInsight API")

# CORS — allow the React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 5.3 Request/Response Models

```python
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # For conversation continuity

class ChatResponse(BaseModel):
    response: str
    session_id: str
```

#### 5.4 The Chat Endpoint — Non-Streaming
The simplest version waits for the full response before returning:

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    result = app_graph.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=request.message)
        ]
    })
    return {"response": result["messages"][-1].content}
```

This works but the user stares at a loading spinner for 5-15 seconds. Not acceptable for a chat interface.

#### 5.5 The Chat Endpoint — Streaming (The Real Implementation)
Streaming sends tokens to the client as the LLM generates them. This is what MarketInsight actually uses.

There are two common approaches:

**Approach A: Server-Sent Events (SSE)**
```python
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        async for event in app_graph.astream_events(
            {"messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=request.message)
            ]},
            version="v2"
        ):
            # Filter for the events we care about
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield f"data: {json.dumps({'token': chunk.content})}\n\n"
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

**Approach B: Newline-Delimited JSON (NDJSON)**
```python
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for event in app_graph.astream_events(
            {"messages": [...]},
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield json.dumps({"token": chunk.content}) + "\n"
        
        yield json.dumps({"done": True}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

#### 5.6 Understanding `astream_events`
LangGraph's `astream_events` emits detailed events for everything happening in the graph. The event types you'll see:

```python
# LLM starts generating
{"event": "on_chat_model_start", "name": "ChatOpenAI", ...}

# Each token from the LLM
{"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content="Apple")}}

# LLM finishes
{"event": "on_chat_model_end", ...}

# Tool execution starts
{"event": "on_tool_start", "name": "get_stock_price", ...}

# Tool execution ends
{"event": "on_tool_end", "data": {"output": "Stock: Apple (AAPL)..."}}
```

For the frontend, you primarily care about `on_chat_model_stream` events from the **final** LLM call (not intermediate reasoning). This requires filtering:

```python
# Only stream tokens from the agent's final response, not tool-calling reasoning
if (event["event"] == "on_chat_model_stream" 
    and event["metadata"].get("langgraph_node") == "agent"):
    # This is a token from the agent node
    chunk = event["data"]["chunk"]
    if chunk.content and not chunk.tool_calls:
        yield f"data: {json.dumps({'token': chunk.content})}\n\n"
```

#### 5.7 Session Management
A real chat app needs conversation history. Users want to ask follow-up questions:
- "What's Apple's P/E ratio?" → agent answers
- "How does that compare to its 5-year average?" → agent needs to know "that" = P/E ratio and "its" = Apple

Options:

**Option A: In-memory sessions (simplest, not production-ready)**
```python
from collections import defaultdict
import uuid

sessions = defaultdict(list)  # session_id → message history

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    # Add user message to session history
    sessions[session_id].append(HumanMessage(content=request.message))
    
    # Build full message list
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + sessions[session_id]
    
    async def generate():
        full_response = ""
        async for event in app_graph.astream_events(
            {"messages": messages}, version="v2"
        ):
            # ... stream tokens ...
            pass
        
        # Save assistant response to session
        sessions[session_id].append(AIMessage(content=full_response))
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Option B: LangGraph checkpointing (production-ready)**
LangGraph has built-in checkpointing that persists the full graph state:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app_graph = graph.compile(checkpointer=memory)

# Each invocation with a thread_id automatically maintains history
config = {"configurable": {"thread_id": session_id}}
result = app_graph.invoke({"messages": [HumanMessage(content=msg)]}, config)
```

#### 5.8 Configuration Management
The `config/` directory in MarketInsight handles:

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

#### 5.9 Error Handling
Production APIs need proper error handling:

```python
from fastapi import HTTPException

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # ... agent execution ...
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
```

### Exercises for This Module
1. Set up a basic FastAPI app with a non-streaming `/chat` endpoint
2. Implement the streaming endpoint using `astream_events`
3. Add session management so follow-up questions work
4. Add CORS middleware configured for the React dev server
5. Test with `curl` or a tool like Postman to verify streaming works
6. Add proper error handling for invalid inputs

### Key Decisions to Discuss Before Coding
- SSE vs NDJSON for streaming format?
- In-memory sessions vs LangGraph checkpointing?
- Should we stream tool call status to the frontend? (e.g., "Fetching Apple stock price...")
- How to handle concurrent requests from the same session?
- Should the API be async all the way through?

---

## Module 6: Langfuse — Observability and Tracing

### What You're Building
Integration with Langfuse that captures every LLM call, tool invocation, and execution trace so you can debug, monitor, and optimize the agent in production.

### Why This Module Exists
Without observability, debugging an AI agent is like debugging code without logs. When a user reports "the agent gave me wrong data about Tesla," you need to know: which tools did it call? What parameters did it use? What did the LLM see? What did it return? Langfuse gives you all of this in a visual dashboard.

### Core Concepts

#### 6.1 What is Langfuse?
Langfuse is an open-source LLM observability platform. It captures **traces** — detailed records of every step in your agent's execution. Think of it as structured logging specifically designed for LLM applications.

A trace in Langfuse shows:
- Every LLM call with full input/output
- Token counts and costs
- Latency for each step
- Tool calls with their arguments and results
- The full conversation context at each point

#### 6.2 Setting Up Langfuse
Langfuse can run self-hosted or as a cloud service:

```python
# Install
# pip install langfuse

from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"  # or your self-hosted URL
)
```

#### 6.3 Integrating with LangGraph
Langfuse integrates via LangChain's callback system. You pass the handler as a callback:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    callbacks=[langfuse_handler]
)

# Every LLM call and tool execution within this invocation gets traced
result = app_graph.invoke(
    {"messages": [...]},
    config=config
)
```

That's it. Every LLM call, tool call, and intermediate step is automatically captured.

#### 6.4 What Gets Traced

For a single user query like "What's Apple's P/E ratio?", Langfuse captures:

```
Trace: "What's Apple's P/E ratio?"
├── Generation: ChatOpenAI (agent node)
│   ├── Input: System prompt + user message
│   ├── Output: AIMessage with tool_calls=[get_financial_ratios("AAPL")]
│   ├── Tokens: 450 input, 35 output
│   ├── Latency: 1.2s
│   └── Cost: $0.0003
├── Span: Tool Execution
│   ├── Tool: get_financial_ratios
│   ├── Input: {"ticker": "AAPL"}
│   ├── Output: "P/E: 28.5, P/B: 45.2, ..."
│   └── Latency: 0.8s
└── Generation: ChatOpenAI (agent node)
    ├── Input: Full history + tool results
    ├── Output: "Apple's current P/E ratio is 28.5..."
    ├── Tokens: 520 input, 150 output
    ├── Latency: 2.1s
    └── Cost: $0.0005
```

#### 6.5 Adding Custom Metadata
You can enrich traces with user info, session IDs, and tags:

```python
langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com",
    session_id=session_id,         # Group traces by conversation
    user_id=user_id,               # Track per-user usage
    tags=["production", "v1.0"],   # Filterable tags
    metadata={"query_type": "financial_ratios"}
)
```

#### 6.6 What to Monitor in Production
Once the agent is live, Langfuse helps you track:

**Performance**:
- Average response latency (target: < 5s for simple queries)
- Token usage per query (cost optimization)
- Tool execution latency (identify slow YFinance calls)

**Quality**:
- Which tools get called most often (optimize those first)
- Tool error rates (broken YFinance endpoints)
- Cases where the LLM calls the wrong tool (improve descriptions)
- Queries that take too many iterations (improve system prompt)

**Cost**:
- Total token consumption per day/week
- Cost per query (helps decide gpt-4o-mini vs gpt-4o)
- Most expensive query patterns

#### 6.7 Debugging with Langfuse
When something goes wrong, Langfuse lets you:
1. Find the specific trace by session ID or time
2. See exactly what the LLM was given as input
3. See what it decided to do (which tools, what arguments)
4. See what the tools returned
5. See how the LLM synthesized the final response

This is invaluable for debugging issues like:
- "The agent said Apple's P/E was 150" → check tool output, was the data wrong from YFinance?
- "The agent called the wrong tool" → check tool descriptions, were they ambiguous?
- "The response took 30 seconds" → check which step was slow (LLM? tool? YFinance?)

#### 6.8 Langfuse vs. Alternatives
Why Langfuse over LangSmith or other options?
- **Open source**: Can self-host for data sovereignty
- **Clean UI**: Easy to navigate traces
- **LangChain native**: Drop-in callback integration
- **Cost tracking**: Built-in token/cost calculations
- **Evaluation support**: Can score traces for quality

LangSmith (by LangChain) is the other major option — tighter LangGraph integration but not open source.

### Exercises for This Module
1. Set up a free Langfuse cloud account
2. Add the CallbackHandler to your LangGraph agent
3. Make 5-10 queries and explore the traces in the dashboard
4. Identify the slowest step in your agent's execution
5. Find a trace where the agent called the wrong tool and analyze why
6. Calculate average cost per query from the dashboard

### Key Decisions to Discuss Before Coding
- Langfuse cloud vs. self-hosted?
- Should we trace in development too, or only production?
- What metadata should we attach to each trace?
- Should we use Langfuse's evaluation features to score responses?

---

## Module 7: React Frontend — Building the Chat Interface

### What You're Building
A React application with a chat interface that connects to the FastAPI backend and displays streaming responses in real-time. This is the `frontend/` directory.

### Why This Module Exists
The backend API is functional but invisible. Users interact through the frontend. The critical engineering challenge here isn't React itself — it's **consuming a streaming API** and rendering tokens as they arrive, which is fundamentally different from typical REST API consumption.

### Core Concepts

#### 7.1 Project Setup
The MarketInsight frontend uses Vite + React + TypeScript:

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

Key dependencies:
- `react-markdown`: Render the LLM's markdown-formatted responses
- `remark-gfm`: GitHub Flavored Markdown support (tables, etc.)

#### 7.2 Application Architecture

```
frontend/
├── src/
│   ├── App.tsx              # Main app component
│   ├── components/
│   │   ├── ChatWindow.tsx   # Message list display
│   │   ├── MessageBubble.tsx # Individual message
│   │   ├── InputBar.tsx     # User input + send button
│   │   └── LoadingIndicator.tsx
│   ├── hooks/
│   │   └── useChat.ts       # Custom hook for chat logic
│   ├── services/
│   │   └── api.ts           # API client with streaming
│   ├── types/
│   │   └── chat.ts          # TypeScript interfaces
│   └── main.tsx
├── index.html
└── package.json
```

#### 7.3 TypeScript Interfaces

```typescript
// types/chat.ts
interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    isStreaming?: boolean;  // True while tokens are still arriving
}

interface ChatState {
    messages: Message[];
    isLoading: boolean;
    sessionId: string | null;
    error: string | null;
}
```

#### 7.4 The Streaming API Client (The Hard Part)
Consuming a streaming API from the browser requires reading a `ReadableStream`:

```typescript
// services/api.ts
export async function streamChat(
    message: string,
    sessionId: string | null,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (error: string) => void
) {
    try {
        const response = await fetch('http://localhost:8000/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) throw new Error('No response body');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value, { stream: true });
            
            // Parse SSE format: "data: {...}\n\n"
            const lines = text.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data.done) {
                        onDone();
                    } else if (data.token) {
                        onToken(data.token);
                    }
                }
            }
        }
    } catch (error) {
        onError(error instanceof Error ? error.message : 'Unknown error');
    }
}
```

**Why this is tricky**:
- The stream arrives in **chunks** that may not align with JSON boundaries
- You need to buffer partial data and parse complete messages
- SSE format has specific delimiters (`data: ` prefix, `\n\n` terminator)
- Error handling must work mid-stream (what if the connection drops?)

#### 7.5 The useChat Hook
Custom React hook that manages chat state and streaming:

```typescript
// hooks/useChat.ts
import { useState, useCallback, useRef } from 'react';
import { streamChat } from '../services/api';
import { Message } from '../types/chat';

export function useChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const streamingContentRef = useRef('');

    const sendMessage = useCallback(async (content: string) => {
        // Add user message
        const userMessage: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content,
            timestamp: new Date(),
        };

        // Add placeholder for assistant response
        const assistantMessage: Message = {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            isStreaming: true,
        };

        setMessages(prev => [...prev, userMessage, assistantMessage]);
        setIsLoading(true);
        streamingContentRef.current = '';

        await streamChat(
            content,
            sessionId,
            // onToken — append each token to the assistant message
            (token) => {
                streamingContentRef.current += token;
                setMessages(prev => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    updated[updated.length - 1] = {
                        ...last,
                        content: streamingContentRef.current,
                    };
                    return updated;
                });
            },
            // onDone
            () => {
                setMessages(prev => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    updated[updated.length - 1] = {
                        ...last,
                        isStreaming: false,
                    };
                    return updated;
                });
                setIsLoading(false);
            },
            // onError
            (error) => {
                setMessages(prev => {
                    const updated = [...prev];
                    const last = updated[updated.length - 1];
                    updated[updated.length - 1] = {
                        ...last,
                        content: `Error: ${error}`,
                        isStreaming: false,
                    };
                    return updated;
                });
                setIsLoading(false);
            }
        );
    }, [sessionId]);

    return { messages, isLoading, sendMessage };
}
```

**Why `useRef` for streaming content?** React state updates are batched. If you do `setMessages` inside every `onToken` callback, React might not re-render for every single token. Using a ref to accumulate content and then updating state ensures you don't lose tokens.

#### 7.6 Rendering Markdown Responses
The LLM returns markdown-formatted text (headers, tables, bold text, lists). You need to render this properly:

```typescript
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function MessageBubble({ message }: { message: Message }) {
    return (
        <div className={`message ${message.role}`}>
            {message.role === 'assistant' ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                </ReactMarkdown>
            ) : (
                <p>{message.content}</p>
            )}
            {message.isStreaming && <span className="cursor">▊</span>}
        </div>
    );
}
```

`remarkGfm` is important — financial data often comes back in tables, and GFM (GitHub Flavored Markdown) is required for table rendering.

#### 7.7 Auto-scrolling
As tokens stream in, the chat should auto-scroll to the bottom:

```typescript
const messagesEndRef = useRef<HTMLDivElement>(null);

useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
}, [messages]);

// In JSX:
<div className="messages">
    {messages.map(msg => <MessageBubble key={msg.id} message={msg} />)}
    <div ref={messagesEndRef} />
</div>
```

#### 7.8 UI States
The interface needs to handle:
- **Idle**: Input enabled, no loading indicator
- **Streaming**: Input disabled, cursor blinking in the response, tokens appearing
- **Error**: Error message displayed, input re-enabled
- **Tool calling**: Optionally show "Analyzing..." or "Fetching data from Yahoo Finance..."

If you stream tool call events from the backend (not just final tokens), you can show intermediate status:

```
User: "Compare Apple and Microsoft"
🔍 Looking up AAPL stock price...
🔍 Looking up MSFT stock price...
📊 Analyzing comparison...
[streaming response appears]
```

#### 7.9 Responsive Design
The repo mentions responsive design for all devices. Key considerations:
- Chat input should be fixed at the bottom (like all chat apps)
- Messages should be scrollable
- On mobile, the input should not be hidden by the keyboard
- Markdown tables might need horizontal scrolling on mobile

### Exercises for This Module
1. Set up the Vite + React + TypeScript project
2. Build the basic chat UI (messages list, input bar)
3. Implement the streaming API client
4. Wire up the useChat hook and test end-to-end with the FastAPI backend
5. Add markdown rendering for assistant responses
6. Add auto-scrolling and loading states
7. Test on mobile viewport

### Key Decisions to Discuss Before Coding
- Should we use a component library (shadcn, MUI) or keep it simple with custom CSS?
- Should tool call status be shown in the UI?
- How to handle markdown rendering during streaming? (Partial markdown can render oddly)
- Should we add a "stop generating" button?
- Dark mode or light mode (or both)?

---

## Module 8: Integration, Deployment, and Polish

### What You're Building
The complete, deployed MarketInsight application — all modules wired together, properly configured, and accessible on the internet.

### Why This Module Exists
Individual modules work in isolation. Integration is where things break. This module covers the last mile: connecting everything, handling production edge cases, and deploying.

### Core Concepts

#### 8.1 End-to-End Integration Checklist

```
□ Frontend sends request to FastAPI
□ FastAPI passes to LangGraph agent
□ Agent reasons and decides on tool calls
□ Tools execute YFinance queries successfully
□ Tool results flow back through the agent
□ Agent synthesizes final response
□ Response streams back through FastAPI
□ Frontend renders streaming tokens
□ Langfuse captures the entire trace
□ Follow-up questions maintain conversation context
□ Error at any step produces a user-friendly message
```

#### 8.2 Environment Configuration

```env
# .env file
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com

FRONTEND_URL=http://localhost:5173
API_HOST=0.0.0.0
API_PORT=8000
```

**Production vs Development**: Use different `.env` files or environment variables for each. Never commit API keys to Git.

#### 8.3 CORS Configuration for Production

```python
# Development
allow_origins=["http://localhost:5173"]

# Production
allow_origins=[
    "https://market-insight-theta.vercel.app",
    "https://yourdomain.com"
]
```

#### 8.4 Deployment Architecture
The MarketInsight repo uses:
- **Frontend**: Vercel (free tier, great for React apps)
- **Backend**: Render (`render.yaml` is present in the repo)

```yaml
# render.yaml
services:
  - type: web
    name: marketinsight-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false  # Set manually in Render dashboard
```

**Alternative deployment options**:
- Railway (simple Python hosting)
- Fly.io (good for low-latency)
- AWS Lambda + API Gateway (serverless, but cold starts hurt streaming)
- Docker on any cloud provider

#### 8.5 Production Concerns

**Rate Limiting**:
- YFinance has unofficial rate limits. Too many concurrent users = blocked.
- OpenAI has token-per-minute limits based on your tier.
- Solution: Add request rate limiting to the API.

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(request: ChatRequest):
    ...
```

**Caching**:
- Stock data doesn't change every second. Cache YFinance responses for 1-5 minutes.
- Company info barely changes — cache for hours.
- Historical data is immutable — cache aggressively.

```python
from functools import lru_cache
import time

_cache = {}
CACHE_TTL = 300  # 5 minutes

def cached_stock_price(ticker: str) -> dict:
    key = f"price:{ticker}"
    if key in _cache and time.time() - _cache[key]["time"] < CACHE_TTL:
        return _cache[key]["data"]
    
    data = yf.Ticker(ticker).info
    _cache[key] = {"data": data, "time": time.time()}
    return data
```

**Error Recovery**:
- If YFinance is down, the tool should return a clear error, and the LLM should tell the user gracefully.
- If the OpenAI API errors, return a fallback message instead of a 500 error.
- If streaming breaks mid-response, the frontend should handle the incomplete message.

**Token Budget Management**:
- Each query costs money. With GPT-4o, a complex multi-tool query might cost $0.01-0.05.
- Track costs via Langfuse and set alerts for unusual spending.
- Consider per-user limits if the app is public.

#### 8.6 Testing Strategy

**Unit Tests**: Test each tool independently
```python
def test_get_stock_price_valid():
    result = get_stock_price.invoke({"ticker": "AAPL"})
    assert "Apple" in result or "AAPL" in result

def test_get_stock_price_invalid():
    result = get_stock_price.invoke({"ticker": "ZZZZZ"})
    assert "error" in result.lower() or "could not" in result.lower()
```

**Integration Tests**: Test the full agent flow
```python
def test_agent_simple_query():
    result = app_graph.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="What's Apple's stock price?")
        ]
    })
    assert result["messages"][-1].content  # Got a response
    assert "AAPL" in str(result["messages"]) or "Apple" in str(result["messages"])
```

**End-to-End Tests**: Test API endpoints
```python
from fastapi.testclient import TestClient

def test_chat_endpoint():
    client = TestClient(app)
    response = client.post("/chat", json={"message": "What's AAPL's price?"})
    assert response.status_code == 200
```

#### 8.7 Performance Optimization
Profile the agent's execution to identify bottlenecks:

1. **LLM latency**: Usually 1-3 seconds per call. Reduce by using `gpt-4o-mini`.
2. **YFinance latency**: Usually 0.5-2 seconds. Reduce with caching.
3. **Multiple tool calls**: If the LLM calls 3 tools sequentially, that's 3 × YFinance latency. Can we parallelize?
4. **Token count**: Large tool outputs mean more input tokens on the next LLM call. Summarize tool outputs.

#### 8.8 Future Enhancements
Once the base system is working, consider:
- **Authentication**: Add user accounts and API keys
- **Persistent sessions**: Store conversations in a database
- **Multiple LLM support**: Let users choose between GPT-4o, Claude, etc.
- **Technical analysis**: Add charting tools (matplotlib → image → display)
- **Portfolio tracking**: Let users save a watchlist
- **News integration**: Combine financial data with news sentiment
- **Alerts**: Notify users when a stock hits a target price

### Exercises for This Module
1. Wire frontend and backend together and test the full flow
2. Deploy the backend to Render and frontend to Vercel
3. Configure production CORS and environment variables
4. Add basic rate limiting
5. Implement caching for at least one tool
6. Set up monitoring in Langfuse for the production deployment

### Key Decisions to Discuss Before Coding
- Render vs Railway vs other hosting for the backend?
- How aggressive should caching be? (Freshness vs. YFinance rate limits)
- Should we add authentication from the start?
- What's the budget for OpenAI API costs during development and production?

---

## Appendix A: Project File Structure

```
MarketInsight/
├── MarketInsight/
│   ├── __init__.py
│   ├── components/          # Module 4: LangGraph agent
│   │   ├── __init__.py
│   │   └── agent.py         # StateGraph definition, nodes, edges
│   └── utils/               # Module 2: LangChain tools
│       ├── __init__.py
│       └── tools.py          # 16 @tool functions
├── config/                   # Module 5: Configuration
│   ├── __init__.py
│   └── settings.py           # Pydantic settings, env vars
├── frontend/                 # Module 7: React app
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── main.py                   # Module 5: FastAPI entry point
├── requirements.txt
├── pyproject.toml
├── render.yaml               # Module 8: Render deployment config
└── .env                      # API keys (never committed)
```

## Appendix B: Dependencies

```
# requirements.txt (estimated based on the stack)
fastapi>=0.104.0
uvicorn>=0.24.0
langchain>=0.2.0
langchain-openai>=0.1.0
langgraph>=0.2.0
langfuse>=2.0.0
yfinance>=0.2.30
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
```

## Appendix C: Recommended Learning Sequence

| Week | Modules | Milestone |
|------|---------|-----------|
| 1 | 1 + 2 | All 16 tools working independently |
| 2 | 3 + 4 | Agent answers stock questions in terminal |
| 3 | 5 + 6 | Streaming API with Langfuse tracing |
| 4 | 7 + 8 | Full app deployed and accessible |

---

> **Next Step**: Let's discuss Module 1 code design together. We'll talk through how to structure the YFinance data layer before writing any code.
