# MarketInsight-AI Financial Analysis Assistant

An AI-powered financial analysis chatbot built with **LangGraph**, **FastAPI**, and **React**. Ask it anything about stocks, company financials, market data, or investing and it will fetch real-time data using a suite of yfinance-backed tools.

## Features

- **Conversational AI agent** powered by GPT-4o-mini via LangGraph's ReAct loop
- **Input guardrail** that rejects off-topic (non-finance) questions before they reach the agent
- **16 financial tools** covering:
  - Stock prices & historical data
  - Balance sheet, income statement, cash flow
  - Financial ratios (P/E, P/B, ROE, margins, …)
  - Dividends, stock splits, earnings
  - Analyst recommendations & price targets
  - Institutional holders, insider transactions
  - Ticker symbol lookup by company name
- **FastAPI** backend with CORS support
- **React + TypeScript + Vite** frontend
- **Observability** via Langfuse and LangSmith

## Project Structure

```
MarketInsight/
├── main.py          # FastAPI app & API endpoints
├── agent.py         # LangGraph agent definition & guardrail
├── tools.py         # 16 yfinance LangChain tools
├── frontend/        # React + TypeScript frontend (Vite)
│   ├── src/
│   ├── public/
│   └── package.json
└── .env.example     # Required environment variables (copy to .env)
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn langchain langgraph langchain-openai \
            yfinance pandas python-dotenv langfuse langchain-langfuse

# Copy and fill in your environment variables
cp .env.example .env
# Edit .env with your actual API keys

# Start the backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Send a finance question, get an AI response |
| `/health` | GET | Health check |

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_DEFAULT_MODEL` | Model to use (default: `gpt-4o-mini`) |
| `MAILTRAP_API_KEY` | Mailtrap API key (for email features) |
| `LANGSMITH_API_KEY` | LangSmith API key (tracing) |
| `LANGSMITH_PROJECT` | LangSmith project name |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key (observability) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |

> **Never commit your `.env` file.** It is listed in `.gitignore`.

## Example Questions

- *"What is Apple's current stock price?"*
- *"Show me Microsoft's income statement"*
- *"What are the analyst recommendations for TSLA?"*
- *"Get the balance sheet for Amazon"*
- *"What's the P/E ratio and profit margin for NVDA?"*
- *"Who are the top institutional holders of Google?"*
