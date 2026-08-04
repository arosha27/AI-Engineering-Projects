"""
main.py
The RAG chat engine. Combines everything you've built so far:
  1. Takes a user's question
  2. Retrieves the most relevant chunks from ChromaDB (same as query.py)
  3. Sends those chunks + the question to Gemini
  4. Returns a natural-language answer, grounded in your documents

Run with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs to test it in your browser.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from google import genai
import joblib
import pandas as pd
import requests
import json

os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

# The n8n webhook URL for escalating low-confidence answers.
# You'll get this URL from n8n in Step 2 below and put it in your .env file.
N8N_ESCALATION_WEBHOOK = os.getenv("N8N_ESCALATION_WEBHOOK", "")

client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_docs")

# ---- Load the churn model + encoders (trained in Part 3) ----
try:
    churn_model = joblib.load("models/churn_model.pkl")
    le_city = joblib.load("models/le_city.pkl")
    le_category = joblib.load("models/le_category.pkl")
    le_coupon = joblib.load("models/le_coupon.pkl")
    CHURN_MODEL_LOADED = True
except FileNotFoundError:
    CHURN_MODEL_LOADED = False
    print("WARNING: Churn model files not found. Run train_churn_model.py first. /predict-churn will be disabled.")

app = FastAPI(title="BrightByte AI Copilot")


# ---- Request/response shapes ----
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confident: bool


class ChurnRequest(BaseModel):
    total_orders: int
    days_since_signup: int
    days_since_last_order: int
    support_tickets: int
    avg_review_rating: float
    total_spent_pkr: float
    city: str
    preferred_category: str
    used_coupon_last_order: str  # "Yes" or "No"


class ChurnResponse(BaseModel):
    churn_risk_score: float
    churn_prediction: str  # "Likely to churn" or "Likely to stay"


# ---- Core RAG logic ----
def retrieve_chunks(question: str, n_results: int = 3):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[question]
    )
    query_embedding = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return chunks, sources


def generate_answer(question: str, chunks: list[str]):
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful customer support assistant for BrightByte Electronics.
Answer the customer's question using ONLY the information in the context below.

Respond with ONLY a JSON object in this exact format, nothing else:
{{"answer": "your answer here", "confident": true or false}}

Set "confident" to false if the context does not clearly contain the answer,
or if you had to guess or infer beyond what's explicitly stated.
If not confident, still give your best attempt at an answer, but be honest that
the customer may need a human to confirm.

CONTEXT:
{context}

CUSTOMER QUESTION:
{question}"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    # Gemini sometimes wraps JSON in ```json fences — strip those before parsing
    raw_text = response.text.strip()
    raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw_text)
        return parsed.get("answer", raw_text), parsed.get("confident", True)
    except json.JSONDecodeError:
        # If parsing fails for any reason, fall back to treating it as low-confidence
        # rather than crashing — a safe default given this feeds an escalation decision
        return raw_text, False


def notify_n8n_escalation(question: str, answer: str, sources: list[str]):
    if not N8N_ESCALATION_WEBHOOK:
        print("N8N_ESCALATION_WEBHOOK not set in .env — skipping escalation notification.")
        return
    try:
        requests.post(
            N8N_ESCALATION_WEBHOOK,
            json={"question": question, "draft_answer": answer, "sources": sources},
            timeout=5
        )
    except requests.exceptions.RequestException as e:
        # Don't let a failed webhook call break the actual user-facing answer
        print(f"Failed to notify n8n escalation webhook: {e}")


# ---- API endpoint ----
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    chunks, sources = retrieve_chunks(request.question)
    answer, confident = generate_answer(request.question, chunks)
    unique_sources = list(set(sources))

    if not confident:
        notify_n8n_escalation(request.question, answer, unique_sources)

    return AskResponse(answer=answer, sources=unique_sources, confident=confident)


@app.get("/customers")
def list_customers():
    """
    Returns all customers with precomputed features, ready to feed into /predict-churn.
    This is what n8n will call to get the list to loop over.
    """
    df = pd.read_csv("data/customers.csv")
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])

    today = pd.Timestamp.now().normalize()
    df["days_since_signup"] = (today - df["signup_date"]).dt.days
    df["days_since_last_order"] = (today - df["last_order_date"]).dt.days

    output_cols = [
        "customer_id", "name", "city", "total_orders",
        "days_since_signup", "days_since_last_order", "support_tickets",
        "avg_review_rating", "total_spent_pkr", "preferred_category",
        "used_coupon_last_order"
    ]
    return df[output_cols].to_dict(orient="records")


@app.post("/predict-churn", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    if not CHURN_MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Churn model not loaded. Run train_churn_model.py first.")

    # Safely encode categorical fields — fall back to 0 if a brand-new category
    # shows up that the model has never seen before (avoids a crash in production)
    def safe_encode(encoder, value):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        return 0

    city_enc = safe_encode(le_city, request.city)
    category_enc = safe_encode(le_category, request.preferred_category)
    coupon_enc = safe_encode(le_coupon, request.used_coupon_last_order)

    input_df = pd.DataFrame([{
        "total_orders": request.total_orders,
        "days_since_signup": request.days_since_signup,
        "days_since_last_order": request.days_since_last_order,
        "support_tickets": request.support_tickets,
        "avg_review_rating": request.avg_review_rating,
        "total_spent_pkr": request.total_spent_pkr,
        "city_enc": city_enc,
        "category_enc": category_enc,
        "coupon_enc": coupon_enc,
    }])

    # Use predicted probability, not just the hard 0/1 label —
    # this is what lets us set a custom risk threshold later in n8n
    risk_score = float(churn_model.predict_proba(input_df)[0][1])
    prediction = "Likely to churn" if risk_score >= 0.5 else "Likely to stay"

    return ChurnResponse(churn_risk_score=round(risk_score, 3), churn_prediction=prediction)


@app.get("/")
def root():
    return {"status": "BrightByte AI Copilot is running. Go to /docs to test the /ask and /predict-churn endpoints."}