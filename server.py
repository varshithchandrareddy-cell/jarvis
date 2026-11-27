import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()
class Request(BaseModel):
    message: str = Field(min_length=1, max_length=30000)

@app.get("/")
def home():
    return FileResponse(Path(__file__).with_name("index.html"))

@app.post("/api/chat")
def chat(request: Request):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "API key not configured")
    try:
        response = OpenAI(timeout=60).responses.create(
            model=os.getenv("JARVIS_MODEL", "gpt-4.1-mini"),
            instructions="You are JARVIS, a concise study and general knowledge assistant. Explain concepts clearly, organize study notes, generate practice quizzes and acknowledge uncertainty. Do not claim to diagnose health conditions or control devices. Give age-appropriate safety guidance. Never claim to have performed external actions.",
            input=request.message,
            max_output_tokens=2000,
        )
        return {"answer": response.output_text}
    except Exception:
        raise HTTPException(502, "AI request failed")
