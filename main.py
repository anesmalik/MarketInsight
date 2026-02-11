from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

langfuse = Langfuse()

from agent import agent, input_guardrail

app = FastAPI(title="MarketInsight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Input guardrail - check before running agent
    if not input_guardrail(request.message):
        return ChatResponse(
            response="I'm a financial analysis assistant. I can only help with questions about stocks, markets, company financials, and investing. Please ask me something finance-related!"
        )

    langfuse_handler = CallbackHandler()

    result = agent.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config={"callbacks": [langfuse_handler]}
    )

    return ChatResponse(response=result["messages"][-1].content)

@app.get("/health")
def health():
    return {"status": "ok"}