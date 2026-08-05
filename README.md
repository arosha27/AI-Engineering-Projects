# BrightByte AI Copilot 🤖

**An enterprise-grade AI system combining Retrieval-Augmented Generation (RAG), predictive churn analytics, and intelligent workflow automation.**

## Project Overview

**Backend live at** : [ai-engineering-projects-production.up.railway.app](https://ai-engineering-projects-production.up.railway.app/docs)

**Frontend live at** : [https://arosha27-ai-engineering-projects-appfrontend-updated-o5jaxf.streamlit.app/](https://bussiness-copilot.streamlit.app/)

BrightByte AI Copilot is a **production-ready system** that demonstrates full-stack AI engineering skills:

- **RAG Chatbot**: Answers customer questions accurately using company documents, with built-in confidence detection
- **Churn Prediction**: XGBoost model predicts high-risk customers with 94.4% accuracy
- **Intelligent Automation**: n8n workflows automatically escalate low-confidence answers and alert on at-risk customers
- **Professional Frontend**: Streamlit dashboard for real-time monitoring and chat

**Use Case**: A customer service system that answers policy questions accurately (using RAG) while simultaneously identifying customers likely to churn (using ML), and automatically escalating uncertain cases to humans.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     BRIGHTBYTE AI COPILOT                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────┐         ┌─────────────────────────┐   │
│  │   FRONTEND LAYER     │         │   MONITORING LAYER      │   │
│  │  (Streamlit)         │         │   (Sidebar Dashboard)   │   │
│  │  - Chat Interface    │         │   - Churn Scores       │   │
│  │  - Real-time Chat    │         │   - Risk Levels        │   │
│  │  - Confidence Badges │         │   - Auto-refresh       │   │
│  └──────────┬───────────┘         └──────────────┬──────────┘   │
│             │                                    │               │
│             │         HTTP REST API              │               │
│             └────────────────────┬────────────────┘               │
│                                  │                               │
│  ┌────────────────────────────────▼──────────────────────────┐  │
│  │          BACKEND LAYER (FastAPI)                          │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  POST /ask            │  GET /customers                   │  │
│  │  └─ RAG Chatbot      │  └─ List all customers            │  │
│  │                       │                                    │  │
│  │  POST /predict-churn  │  GET /docs                        │  │
│  │  └─ Churn scoring    │  └─ Interactive API docs          │  │
│  └──────┬─────────────────────────────────────────┬──────────┘  │
│         │                                          │             │
│  ┌──────▼────────────────┐              ┌────────▼──────────┐   │
│  │  RAG PIPELINE         │              │  ML PIPELINE      │   │
│  ├──────────────────────┤              ├───────────────────┤   │
│  │                      │              │                   │   │
│  │ 1. CHUNKING          │              │ 1. FEATURE ENG.   │   │
│  │    (Text Splitting)  │              │    (Dates→Days)   │   │
│  │                      │              │                   │   │
│  │ 2. EMBEDDINGS        │              │ 2. MODEL TRAIN    │   │
│  │    (Gemini API)      │              │    (XGBoost)      │   │
│  │                      │              │    (Scored: 0.94) │   │
│  │ 3. VECTOR DB         │              │                   │   │
│  │    (ChromaDB)        │              │ 3. PREDICTION     │   │
│  │    (8 chunks stored) │              │    (Risk scores)  │   │
│  │                      │              │                   │   │
│  │ 4. RETRIEVAL         │              │ 4. MLOPS          │   │
│  │    (Semantic Search) │              │    (MLflow track) │   │
│  │                      │              │                   │   │
│  │ 5. LLM GENERATION    │              └───────────────────┘   │
│  │    (Gemini 3.6)      │                                       │
│  │ + Confidence Check   │                                       │
│  │    (JSON prompt)     │                                       │
│  └──────┬───────────────┘                                       │
│         │                                                        │
│         │        ESCALATION TRIGGER                             │
│         │        (Low confidence OR                             │
│         │         High churn risk)                              │
│         │                                                        │
│  ┌──────▼───────────────────────────────────────────────────┐   │
│  │  AUTOMATION LAYER (n8n)                                  │   │
│  ├───────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Workflow 1: SCHEDULE-TRIGGERED CHURN ALERTS           │   │
│  │  ├─ Daily schedule (or manual trigger)                │   │
│  │  ├─ Fetch 300 customers                               │   │
│  │  ├─ Score each for churn (parallelized)               │   │
│  │  ├─ IF risk >= 0.7: Send alert                        │   │
│  │  └─ Output: Webhook alerts (ready for Slack)          │   │
│  │                                                         │   │
│  │  Workflow 2: WEBHOOK-TRIGGERED ESCALATIONS             │   │
│  │  ├─ Receives low-confidence questions from FastAPI    │   │
│  │  ├─ Logs question + draft answer                      │   │
│  │  ├─ Ready to forward to Slack/support ticket          │   │
│  │  └─ Visible in n8n Executions tab                     │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  DATA STORES:                                                   │
│  • ChromaDB: 8 document chunks (policies, FAQs)                │
│  • XGBoost model: 9-feature churn predictor                    │
│  • MLflow: Model training experiments & metrics                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Features

### 1. **Retrieval-Augmented Generation (RAG) Chatbot**
- Ingests company documents (return policy, shipping, warranty, FAQs)
- Chunks documents intelligently (~300-word chunks with overlap)
- Generates embeddings using **Google Gemini API** (768-dim vectors)
- Stores in **ChromaDB** for semantic search
- Retrieves top-3 relevant chunks per query
- Gemini LLM generates grounded answers (never hallucinates outside docs)
- **Self-rates confidence** — marks low-confidence answers for human review

**Why it works**: Semantic embeddings find the *right piece* of a long document, even with paraphrased questions. Chunking+overlap ensures complete answers, not fragments.

### 2. **Predictive Churn Model**
- Trained on 300 synthetic customers with realistic churn patterns
- Features: recency, frequency, monetary value, support tickets, review ratings
- **Model comparison**: Logistic Regression, Random Forest, XGBoost
- **Winner**: Logistic Regression (F1=0.571) — simple models often outperform complex ones on noisy data
- Real-time scoring via `/predict-churn` endpoint
- Returns **probability scores**, not just 0/1 labels (enables custom thresholds)

**Why it matters**: Early churn detection lets companies intervene before customers leave.

### 3. **Intelligent Automation (n8n)**
Two complementary workflows:

**A) Schedule-Triggered Churn Alerts**
- Runs daily (configurable)
- Fetches all customers
- Scores each in parallel (~30 seconds for 300)
- Fires alerts for risk ≥ 0.7
- Ready to push to Slack/email

**B) Webhook-Triggered Escalations**
- FastAPI directly calls n8n when confidence drops
- Question + draft answer logged
- Human reviews in n8n dashboard
- Prevents hallucinated answers reaching customers

### 4. **Professional Streamlit Dashboard**
- **Cream/beige professional theme** (not default dark mode)
- Real-time chat with source citations
- Sidebar risk dashboard (refresh to score all 300 customers)
- Confidence badges (🟢 Confident / 🟡 Low Confidence)
- Responsive error handling (graceful failures)

---

## 📊 Model Performance

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 97.8% | **0.571** | 0.47 | 0.73 |
| Random Forest | 97.8% | 0.571 | 0.40 | 1.00 |
| XGBoost | 94.4% | 0.800 | 1.00 | 0.67 |

**Selected Model**: Logistic Regression  
**Rationale**: Best F1 on imbalanced data (18.7% churn rate). Recall of 0.73 = catches 73% of churners. False-positive cost (unnecessary retention offer) is lower than false-negative cost (missed churner).

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- Docker (for n8n)
- Google Gemini API key (free tier)

### 1. Clone & Setup
```bash
git clone https://github.com/arosha27/business-copilot.git
cd business-copilot
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env` file:
```
GEMINI_API_KEY=your_key_here
N8N_ESCALATION_WEBHOOK=http://localhost:5678/webhook/brightbyte-escalation
```

Get Gemini API key: https://ai.google.dev

### 3. Ingest Documents & Train Model
```bash
# Ingest company documents into ChromaDB (one-time)
python app/ingest.py

# Train churn model with MLflow tracking
python app/train_churn_model.py

# View training experiments
mlflow ui  # http://127.0.0.1:5000
```

### 4. Start Backend (FastAPI)
```bash
uvicorn app.main:app --reload
# API available at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

### 5. Start Frontend (Streamlit) — in a NEW terminal
```bash
streamlit run app/frontend.py
# Dashboard available at http://127.0.0.1:8501
```

### 6. (Optional) Set Up n8n Automations
```bash
docker run -it --rm -p 5678:5678 n8nio/n8n
# n8n available at http://localhost:5678
# Import workflows: n8n_churn_alert_workflow.json, n8n_escalation_workflow.json
```

---

## 📁 Project Structure

```
business-copilot/
├── app/
│   ├── main.py                  # FastAPI backend (RAG + churn + webhooks)
│   ├── frontend.py              # Streamlit dashboard
│   ├── ingest.py                # Document chunking → ChromaDB
│   ├── train_churn_model.py     # XGBoost training + MLflow
│   ├── query.py                 # Test RAG retrieval (debugging)
│   └── __init__.py
├── data/
│   ├── docs/                    # Company documents (4 policies + FAQ)
│   │   ├── return_policy.txt
│   │   ├── shipping_policy.txt
│   │   ├── warranty_policy.txt
│   │   └── faq.txt
│   └── customers.csv            # 300 synthetic customers (churn labels)
├── models/
│   ├── churn_model.pkl          # Trained XGBoost (saved after training)
│   ├── le_city.pkl              # Label encoder (categorical features)
│   ├── le_category.pkl
│   ├── le_coupon.pkl
│   └── best_model_name.txt      # Which model won the comparison
├── chroma_db/                   # ChromaDB vector store (auto-created)
│   └── ...
├── mlruns/                      # MLflow experiment tracking (auto-created)
│   └── ...
├── n8n_churn_alert_workflow.json       # n8n workflow: schedule → churn alerts
├── n8n_escalation_workflow.json        # n8n workflow: webhook → escalate low-conf
├── requirements.txt             # Python dependencies (pinned versions)
├── .env                         # API keys (GITIGNORE THIS)
├── .gitignore                   # Git exclusions
├── .streamlit/
│   └── config.toml              # Streamlit theme config (cream/beige)
└── README.md                    # This file
```

---

## 🔧 Technology Stack

| Layer | Technology | Why |
|---|---|---|
| **Frontend** | Streamlit 1.41.1 | Fast prototyping, reactive UI, professional theming |
| **Backend API** | FastAPI 0.115.6 | Production-grade, async, OpenAPI docs auto-generated |
| **RAG Pipeline** | ChromaDB 0.5.23, Gemini API | Vector DB stability, free LLM access |
| **ML/Churn** | XGBoost 2.1.3, Scikit-learn 1.5.2 | Gradient boosting + classic ML baseline |
| **ML Tracking** | MLflow 2.19.0 | Experiment comparison, model versioning |
| **Automation** | n8n (Docker) | Visual workflows, webhook triggers, integrations |
| **Language** | Python 3.10 | Data science ecosystem, FastAPI support |

---

## 🧪 Testing the System

### Test 1: RAG Confidence Detection
```bash
# Question IN scope (should be confident)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long do I have to return an item?"}'
# Response: confident = true

# Question OUT of scope (should escalate)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Do you sell electric scooters?"}'
# Response: confident = false, triggers n8n webhook
```

### Test 2: Churn Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict-churn \
  -H "Content-Type: application/json" \
  -d '{
    "total_orders": 2,
    "days_since_signup": 200,
    "days_since_last_order": 150,
    "support_tickets": 3,
    "avg_review_rating": 2.1,
    "total_spent_pkr": 8000,
    "city": "Karachi",
    "preferred_category": "Smartphones",
    "used_coupon_last_order": "No"
  }'
# Response: churn_risk_score = 0.743, churn_prediction = "Likely to churn"
```

### Test 3: Streamlit Dashboard
1. Open http://127.0.0.1:8501
2. Click "Refresh Customer Data" (loads & scores 300 customers)
3. Chat works immediately
4. Low-confidence questions trigger n8n escalations (visible in n8n dashboard)

---

## 🎓 What This Project Demonstrates

### For AI/ML Engineer Interviews:
✅ **RAG Architecture**: Chunking, embeddings, retrieval, LLM integration  
✅ **ML Pipeline**: Feature engineering, model comparison, MLflow tracking  
✅ **Production Code**: Error handling, API design, database integration  
✅ **Automation**: Workflow orchestration, webhook handling, scheduling  
✅ **Full Stack**: Backend API, frontend UI, data pipeline, monitoring  
✅ **Problem Solving**: Confidence detection, imbalanced data handling, graceful degradation  

### Interview Questions You'll Be Ready For:
- *"Walk me through your RAG system"* → Explain the architecture diagram
- *"Why XGBoost over Random Forest?"* → Show the model comparison table
- *"How do you handle low-quality LLM outputs?"* → Confidence detection + n8n escalation
- *"What would you do in production?"* → Describe Slack integration, monitoring, retrain loops

---

## 🔮 Future Enhancements (Portfolio Talking Points)

1. **Slack Integration** — Direct alerts to support channel instead of n8n logs
2. **Feedback Loop** — User feedback on answers retrains the churn model
3. **Multi-Language RAG** — Support Urdu/Hindi for Pakistan market
4. **A/B Testing** — Compare RAG responses vs. rule-based baseline
5. **Real Database** — Replace CSV with PostgreSQL for production scale
6. **LLM Fine-Tuning** — Fine-tune Gemini on company-specific FAQs
7. **Monitoring Dashboard** — Grafana + Prometheus for uptime tracking
8. **Docker Compose** — One-command deployment (FastAPI + ChromaDB + n8n)

---

## 📝 License

MIT License — See LICENSE file for details

---

## 👤 Author

**Arosha Bakhtawar**  
Data Science + AI Engineer  
📧 aroshaamin0@gmail.com  
🔗 LinkedIn: linkedin.com/in/arosha-amin/  
🔗 GitHub: github.com/arosha27  
🔗 Kaggle: kaggle.com/aroshabakhtawar

---

## 🙏 Acknowledgments

- **Google Gemini API** for free-tier LLM access
- **n8n** for visual workflow automation
- **Streamlit** for rapid UI development
- **ChromaDB** for lightweight vector storage

---

**Last Updated**: August 2026  
**Status**: Production-ready for portfolio demonstration
