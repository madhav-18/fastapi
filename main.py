from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import rag_response
from app.classification import classify_text
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

app = FastAPI()

class RAGInput(BaseModel):
    prompt: str

class ClassificationInput(BaseModel):
    text: str

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Chatbot API"}

@app.post("/rag")
@cache(expire=60)
async def rag_endpoint(input_data: RAGInput):
    return rag_response(input_data.prompt)

@app.post("/classification")
@cache(expire=60)
async def classification_endpoint(input_data: ClassificationInput):
    return classify_text(input_data.text)