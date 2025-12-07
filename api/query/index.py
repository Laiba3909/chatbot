# api/query/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from embedding import generate_answer
from mangum import Mangum  # Adapter for serverless

app = FastAPI(title="Book Chatbot API")

class Query(BaseModel):
    question: str

@app.post("/")
def chat(query: Query):
    try:
        answer = generate_answer(query.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

# Vercel serverless handler
handler = Mangum(app)

