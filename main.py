from pydantic import BaseModel
from search import agent
from fastapi import FastAPI

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "HxH lore agent is alive"}

@app.post("/ask")
def ask(request: AskRequest):
    response = agent.invoke({"messages": [("user", request.question)]})
    answer = response["messages"][-1].content
    return {"answer": answer}