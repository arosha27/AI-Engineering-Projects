"""
ingest.py
Reads all .txt documents from data/docs, splits them into chunks,
turns each chunk into an embedding using Gemini, and stores them in ChromaDB.

Run this once (or whenever your documents change):
    python app/ingest.py
"""

import os
import glob
from dotenv import load_dotenv
import chromadb
from google import genai

# ---- 1. Load your API key from .env ----
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

# ---- 2. Set up ChromaDB (saves to disk in chroma_db/ folder) ----
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_docs")

# ---- 3. Simple chunking function ----
def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping word chunks.
    Overlap helps avoid cutting an answer in half between two chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---- 4. Read all documents and chunk them ----
doc_files = glob.glob("data/docs/*.txt")
print(f"Found {len(doc_files)} documents.")

all_chunks = []
all_ids = []
all_metadatas = []

for file_path in doc_files:
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"  {file_name}: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{file_name}_chunk_{i}")
        all_metadatas.append({"source": file_name, "chunk_index": i})

print(f"\nTotal chunks to embed: {len(all_chunks)}")

# ---- 5. Generate embeddings for each chunk (batched) ----
print("Generating embeddings with Gemini...")

embeddings = []
batch_size = 10  # keep batches small to avoid rate limits on free tier

for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i + batch_size]
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch
    )
    for emb in result.embeddings:
        embeddings.append(emb.values)
    print(f"  Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")

# ---- 6. Store everything in ChromaDB ----
collection.add(
    ids=all_ids,
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=all_metadatas
)

print(f"\nDone. Stored {len(all_chunks)} chunks in ChromaDB collection 'company_docs'.")
print("You can now run app/query.py to test retrieval.")