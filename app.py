from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import os

from src.llama_pinecone import generate_response as generate_response_llama
# TODO: Import Falcon generate_response when implemented
# from src.falcon_chrome import generate_response as generate_response_falcon

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    model: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.model == "openai":
        answer = "Please use ai7.breezemytrip.com for OpenAI"
    elif request.model == "llama":
        answer = generate_response_llama(request.message)
    elif request.model == "falcon":
        # TODO: Implement Falcon response
        answer = "Falcon model not yet implemented"
    else:
        answer = "Invalid model selection"
    return {"response": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
