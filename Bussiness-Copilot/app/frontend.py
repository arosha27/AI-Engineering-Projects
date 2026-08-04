"""
Streamlit Frontend for BrightByte AI Copilot
A professional, production-ready chat and monitoring interface
for the RAG + churn prediction + automation system.

Run with:
    streamlit run app/frontend.py
"""

import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
from typing import Optional

# Page configuration with light theme
st.set_page_config(
    page_title="BrightByte AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Cream/Beige Color Palette ---
PRIMARY_COLOR = "#2C3E50"        # Dark blue-gray (professional)
SECONDARY_COLOR = "#E8DCC4"      # Cream/beige
ACCENT_COLOR = "#D4A574"         # Warm taupe
SUCCESS_COLOR = "#5F9B6B"        # Muted green
WARNING_COLOR = "#C9915A"        # Warm amber
DANGER_COLOR = "#A0616A"         # Muted rose
NEUTRAL_LIGHT = "#F5F3F0"        # Off-white/cream
NEUTRAL_DARK = "#4A4542"         # Charcoal

# Apply theme via Streamlit config
st.markdown(f"""
    <style>
    /* Root color variables */
    :root {{
        --primary: {PRIMARY_COLOR};
        --secondary: {SECONDARY_COLOR};
        --accent: {ACCENT_COLOR};
        --success: {SUCCESS_COLOR};
        --warning: {WARNING_COLOR};
        --danger: {DANGER_COLOR};
        --light: {NEUTRAL_LIGHT};
        --dark: {NEUTRAL_DARK};
    }}
    
    /* Main background */
    .stApp {{
        background-color: {NEUTRAL_LIGHT};
        color: {NEUTRAL_DARK};
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #4A4542;
        border-right: 2px solid {SECONDARY_COLOR};
    }}
    
    /* Chat message styling */
    .user-message {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%);
        color: white;
        padding: 14px 16px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid {ACCENT_COLOR};
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.1);
    }}
    
    .bot-message {{
        background-color: #FFFFFF;
        color: {NEUTRAL_DARK};
        padding: 14px 16px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid {SUCCESS_COLOR};
        box-shadow: 0 2px 8px rgba(95, 155, 107, 0.1);
    }}
    
    .confidence-low {{
        background-color: #FEF5F1;
        border-left-color: {DANGER_COLOR};
    }}
    
    /* Metric cards */
    .metric-card {{
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(44, 62, 80, 0.15);
        font-weight: 500;
    }}
    
    .churn-risk-high {{
        background: linear-gradient(135deg, {DANGER_COLOR} 0%, #8B5A5A 100%);
    }}
    
    .churn-risk-medium {{
        background: linear-gradient(135deg, {WARNING_COLOR} 0%, #D4B5A0 100%);
    }}
    
    .churn-risk-low {{
        background: linear-gradient(135deg, {SUCCESS_COLOR} 0%, #7AAD80 100%);
    }}
    
    /* Header styling */
    .header-title {{
        font-size: 36px;
        font-weight: 700;
        color: white;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }}
    
    .header-subtitle {{
        font-size: 15px;
        color: {NEUTRAL_DARK};
        margin-bottom: 20px;
        opacity: 0.8;
    }}
    
    /* Status badges */
    .status-badge {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 4px 4px 0;
        letter-spacing: 0.3px;
    }}
    
    .badge-success {{
        background-color: #E8F4ED;
        color: {SUCCESS_COLOR};
        border: 1px solid {SUCCESS_COLOR};
    }}
    
    .badge-warning {{
        background-color: #FFF0E6;
        color: {WARNING_COLOR};
        border: 1px solid {WARNING_COLOR};
    }}
    
    .badge-danger {{
        background-color: #F5E8E6;
        color: {DANGER_COLOR};
        border: 1px solid {DANGER_COLOR};
    }}
    
    /* Source citations */
    .source-citation {{
        background-color: {SECONDARY_COLOR};
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 13px;
        color: {NEUTRAL_DARK};
        margin: 6px 0;
        border-left: 4px solid {ACCENT_COLOR};
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%);
        color: black;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        box-shadow: 0 4px 12px rgba(44, 62, 80, 0.3);
        transform: translateY(-2px);
    }}
    
    /* Input field placeholder styling */
    .stTextInput > div > div > input::placeholder {{
        color: black;
        opacity: 0.6;
    }}
    
    /* Input styling */
    .stTextInput > div > div > input {{
        background-color: #FFFFFF;
        border: 2px solid {SECONDARY_COLOR};
        color: black;
        padding: 12px;
        border-radius: 8px;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {ACCENT_COLOR};
        box-shadow: 0 0 0 3px rgba(212, 165, 116, 0.1);
    }}
    
    /* Divider */
    .stDivider {{
        border-color: {SECONDARY_COLOR};
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: {SECONDARY_COLOR};
        color: {NEUTRAL_DARK};
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {{
        color: {NEUTRAL_DARK};
        border-bottom: 3px solid transparent;
    }}
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom-color: {ACCENT_COLOR};
        color: {PRIMARY_COLOR};
        font-weight: 600;
    }}
    
    /* Info/Warning/Error boxes */
    .stInfo {{
        background-color: #E8F4ED;
        border-left-color: {SUCCESS_COLOR};
        color: black;
    }}
    
    .stInfo > div {{
        color: black;
    }}
    
    .stWarning {{
        background-color: #FFF0E6;
        border-left-color: {WARNING_COLOR};
        color: black;
    }}
    
    .stError {{
        background-color: #F5E8E6;
        border-left-color: {DANGER_COLOR};
        color: black;
    }}
    
    .stSuccess {{
        background-color: #E8F4ED;
        border-left-color: {SUCCESS_COLOR};
        color: black;
    }}
    
    /* Spinner styling */
    .stSpinner {{
        color: {ACCENT_COLOR};
    }}
    </style>
""", unsafe_allow_html=True)

# --- API Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "churn_data" not in st.session_state:
    st.session_state.churn_data = None

if "last_escalation" not in st.session_state:
    st.session_state.last_escalation = None

# --- Helper Functions ---

def safe_api_call(endpoint: str, method: str = "GET", json_data: Optional[dict] = None) -> Optional[dict]:
    """
    Safely call the FastAPI backend with error handling.
    Returns the JSON response or None if failed.
    """
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        elif method == "POST":
            response = requests.post(url, json=json_data, timeout=REQUEST_TIMEOUT)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. The backend may be down. Check if uvicorn is running.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend at http://127.0.0.1:8000. Start uvicorn with: `uvicorn app.main:app --reload`")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


def ask_question(question: str) -> tuple[Optional[str], Optional[list], Optional[bool]]:
    """
    Send a question to the RAG endpoint and get back answer, sources, confidence.
    Returns (answer, sources, confident) or (None, None, None) on failure.
    """
    response = safe_api_call("/ask", method="POST", json_data={"question": question})
    
    if response:
        return (
            response.get("answer"),
            response.get("sources", []),
            response.get("confident", True)
        )
    return None, None, None


def predict_churn_for_customer(customer_data: dict) -> Optional[dict]:
    """
    Predict churn risk for a single customer.
    Returns {churn_risk_score, churn_prediction} or None on failure.
    """
    response = safe_api_call("/predict-churn", method="POST", json_data=customer_data)
    return response


def fetch_all_customers() -> Optional[list]:
    """
    Fetch all customers with their features for churn scoring.
    """
    response = safe_api_call("/customers", method="GET")
    return response if isinstance(response, list) else None


def score_all_customers_for_churn(customers: list) -> pd.DataFrame:
    """
    Score all customers and return a DataFrame with churn risks sorted highest-first.
    """
    high_risk = []
    
    for customer in customers:
        # Extract required fields for prediction
        prediction_data = {
            "total_orders": customer.get("total_orders", 0),
            "days_since_signup": customer.get("days_since_signup", 0),
            "days_since_last_order": customer.get("days_since_last_order", 0),
            "support_tickets": customer.get("support_tickets", 0),
            "avg_review_rating": customer.get("avg_review_rating", 3.0),
            "total_spent_pkr": customer.get("total_spent_pkr", 0),
            "city": customer.get("city", "Karachi"),
            "preferred_category": customer.get("preferred_category", "Smartphones"),
            "used_coupon_last_order": customer.get("used_coupon_last_order", "No"),
        }
        
        score = predict_churn_for_customer(prediction_data)
        
        if score:
            high_risk.append({
                "customer_id": customer.get("customer_id"),
                "name": customer.get("name"),
                "city": customer.get("city"),
                "total_orders": customer.get("total_orders"),
                "days_since_last_order": customer.get("days_since_last_order"),
                "churn_risk": score["churn_risk_score"],
                "prediction": score["churn_prediction"],
            })
    
    df = pd.DataFrame(high_risk)
    if not df.empty:
        df = df.sort_values("churn_risk", ascending=False)
    
    return df


def get_risk_level_badge(risk_score: float) -> str:
    """Return HTML badge based on risk score."""
    if risk_score >= 0.7:
        return f'<span class="status-badge badge-danger">HIGH RISK ({risk_score:.1%})</span>'
    elif risk_score >= 0.5:
        return f'<span class="status-badge badge-warning">MEDIUM RISK ({risk_score:.1%})</span>'
    else:
        return f'<span class="status-badge badge-success">LOW RISK ({risk_score:.1%})</span>'


# --- Main Layout ---

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="header-title">🤖 BrightByte AI Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Intelligent Q&A + Customer Risk Monitoring</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 20px;">
        <small>Last refresh: {datetime.now().strftime('%H:%M:%S')}</small>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Sidebar: Customer Risk Monitoring ---
with st.sidebar:
    st.markdown("### 📊 Customer Risk Dashboard")
    
    if st.button("🔄 Refresh Customer Data", use_container_width=True):
        with st.spinner("Loading customer data..."):
            customers = fetch_all_customers()
            if customers:
                st.session_state.churn_data = score_all_customers_for_churn(customers)
                st.success(f"✅ Scored {len(customers)} customers")
            else:
                st.error("Failed to fetch customer data")
    
    st.divider()
    
    if st.session_state.churn_data is not None and not st.session_state.churn_data.empty:
        df = st.session_state.churn_data
        
        # Summary metrics
        high_risk_count = len(df[df["churn_risk"] >= 0.7])
        medium_risk_count = len(df[(df["churn_risk"] >= 0.5) & (df["churn_risk"] < 0.7)])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 High Risk", high_risk_count)
        with col2:
            st.metric("🟡 Medium Risk", medium_risk_count)
        
        st.divider()
        
        # Tabs for different risk levels
        tab1, tab2, tab3 = st.tabs(["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"])
        
        with tab1:
            high_risk_df = df[df["churn_risk"] >= 0.7]
            if not high_risk_df.empty:
                for idx, row in high_risk_df.head(5).iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="metric-card churn-risk-high">
                            <strong>{row['name']}</strong> ({row['customer_id']})<br>
                            <small>{row['city']} • {row['total_orders']} orders • Last order {row['days_since_last_order']}d ago</small><br>
                            <strong>Risk: {row['churn_risk']:.1%}</strong>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No high-risk customers at this time")
        
        with tab2:
            medium_risk_df = df[(df["churn_risk"] >= 0.5) & (df["churn_risk"] < 0.7)]
            if not medium_risk_df.empty:
                for idx, row in medium_risk_df.head(5).iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="metric-card churn-risk-medium">
                            <strong>{row['name']}</strong> ({row['customer_id']})<br>
                            <small>{row['city']} • {row['total_orders']} orders</small><br>
                            <strong>Risk: {row['churn_risk']:.1%}</strong>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No medium-risk customers")
        
        with tab3:
            low_risk_df = df[df["churn_risk"] < 0.5]
            st.success(f"✅ {len(low_risk_df)} customers are stable (low risk)")
    
    else:
        st.info("Click 'Refresh Customer Data' to load and score all customers")

# --- Main Content: Chat Interface ---
st.markdown("### 💬 Ask BrightByte")
st.markdown("Ask questions about policies, shipping, returns, warranties, and more.")

# Chat history display
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        confidence_class = "" if message.get("confident", True) else " confidence-low"
        confidence_badge = "🟢 Confident" if message.get("confident", True) else "🟡 Low Confidence"
        
        st.markdown(f'<div class="bot-message"><strong>Assistant:</strong> {confidence_class}\n{message["content"]}<br><small>{confidence_badge}</small></div>', unsafe_allow_html=True)
        
        # Display sources if available
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f'<div class="source-citation">📎 {source}</div>', unsafe_allow_html=True)

st.divider()

# Input area
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Your question:",
        placeholder="E.g., 'What's your return policy?' or 'Do you offer cash on delivery?'",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", use_container_width=True, type="primary")

# Process user input
if send_button and user_input.strip():
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Get response from backend
    with st.spinner("🤔 Thinking..."):
        answer, sources, confident = ask_question(user_input)
    
    if answer:
        # Track low-confidence escalations
        if not confident:
            st.session_state.last_escalation = {
                "question": user_input,
                "answer": answer,
                "timestamp": datetime.now(),
            }
        
        # Add bot response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confident": confident,
        })
        
        # Show notification for escalations
        if not confident:
            st.warning("⚠️ This answer was flagged as low confidence and has been escalated for human review via n8n.")
        
        # Rerun to display new message
        st.rerun()
    else:
        st.error("Failed to get response from backend. Make sure the API is running.")

# --- Footer ---
st.divider()
st.markdown(f"""
<div style="text-align: center; color: {NEUTRAL_DARK}; font-size: 12px; padding: 20px 0; opacity: 0.75;">
    <p style="margin: 8px 0; font-weight: 500;">BrightByte AI Copilot v1.0</p>
    <p style="margin: 4px 0; font-size: 11px;">Enterprise RAG • Predictive Analytics • Intelligent Automation</p>
    <p style="margin: 8px 0; font-size: 10px; letter-spacing: 0.5px;">
        <strong>Powered by:</strong> FastAPI • ChromaDB • Google Gemini • n8n Workflows
    </p>
</div>
""", unsafe_allow_html=True)