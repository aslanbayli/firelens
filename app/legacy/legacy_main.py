from pydantic import BaseModel
from fastapi import FastAPI
from app.services.llm import run_conversation

app = FastAPI(title="firelens")


class Prompt(BaseModel):
    prompt: str


@app.post("/")
def create_prompt(prompt: Prompt):
    return run_conversation(prompt.prompt)
