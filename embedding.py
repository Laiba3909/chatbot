# embeddings.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Qdrant client
qdrant = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

# OpenAI client
openai = OpenAI(api_key=GEMINI_API_KEY)

def search_similar_docs(query, top_k=3):
    # Generate embedding for query
    embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )['data'][0]['embedding']

    # Query Qdrant for similar vectors
    result = qdrant.search(
        collection_name="docs",  # your collection
        query_vector=embedding,
        limit=top_k
    )

    # Return text from top results
    texts = [hit.payload['text'] for hit in result]
    return texts

def generate_answer(query):
    # Retrieve relevant docs
    context = search_similar_docs(query)
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"

    response = openai.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content
