"""
query.py
A simple test script to confirm your ingestion worked.
Type a question, and it shows you the most relevant chunks retrieved from ChromaDB.

This does NOT call the LLM yet — it only tests retrieval (Step 2.3 in the guide).
Run:
    python app/query.py
"""

import os
from dotenv import load_dotenv
import chromadb
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_docs")

print(f"Collection currently has {collection.count()} chunks stored.\n")

question = input("Ask a question about BrightByte's policies: ")

# Embed the question the same way we embedded the chunks
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[question]
)
query_embedding = result.embeddings[0].values

# Search ChromaDB for the most similar chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\n--- Top matching chunks ---\n")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"[{i+1}] Source: {meta['source']} (chunk {meta['chunk_index']})")
    print(doc[:300] + ("..." if len(doc) > 300 else ""))
    print()