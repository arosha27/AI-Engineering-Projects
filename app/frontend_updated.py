"""
Streamlit Frontend for BrightByte AI Copilot
A professional, production-ready chat and monitoring interface
for the RAG + churn prediction + automation system.

Run with:
    streamlit run app/frontend_updated.py
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
    initial_sidebar_state="collapsed"
)

# --- Red/White/Black Color Palette ---
PRIMARY_COLOR = "#DC2626"       # Red (primary accent)
SECONDARY_COLOR = "#1F1F1F"      # Dark gray/black
ACCENT_COLOR = "#EF4444"         # Bright red
SUCCESS_COLOR = "#16A34A"        # Green
WARNING_COLOR = "#F59E0B"        # Amber
DANGER_COLOR = "#DC2626"         # Red
NEUTRAL_LIGHT = "#FFFFFF"        # White
NEUTRAL_DARK = "#000000"         # Black

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
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {{
        display: none;
    }}
    
    /* Chat message styling */
    .user-message {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%);
        color: white;
        padding: 14px 16px;
        border-radius: 12px;
        margin: 5px 0;
        border-left: 4px solid {ACCENT_COLOR};
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.2);
    }}
    
    .bot-message {{
        background-color: #F5F5F5;
        color: {NEUTRAL_DARK};
        padding: 14px 16px;
        border-radius: 12px;
        margin: 5px 0;
        border-left: 4px solid {PRIMARY_COLOR};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .confidence-low {{
        background-color: #FEF2F2;
        border-left-color: {DANGER_COLOR};
    }}
    
    /* Metric cards */
    .metric-card {{
        color: white;
   
        padding: 10px;
        border-radius: 12px;
        margin: 5px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-weight: 500;
    }}
    
    .churn-risk-high {{
        background: linear-gradient(135deg, {DANGER_COLOR} 0%, #991B1B 100%);
    }}
    
    .churn-risk-medium {{
        background: linear-gradient(135deg, {WARNING_COLOR} 0%, #B45309 100%);
    }}
    
    .churn-risk-low {{
        background: linear-gradient(135deg, {SUCCESS_COLOR} 0%, #15803D 100%);
    }}
    
    /* Header styling */
    .header-title {{
        font-size: 36px;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }}
    
    .header-subtitle {{
        font-size: 15px;
        color: {NEUTRAL_DARK};
        margin-bottom: 10px;
        opacity: 0.8;
    }}
    
    /* Status badges */
    .status-badge {{
        display: inline-block;
        color:black;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px 4px 4px 0;
        letter-spacing: 0.3px;
    }}
    
    .badge-success {{
        background-color: {SUCCESS_COLOR};
        color: white;
        border: 1px solid {SUCCESS_COLOR};
    }}
    
    .badge-warning {{
        background-color: {WARNING_COLOR};
        color: white;
        border: 1px solid {WARNING_COLOR};
    }}
    
    .badge-danger {{
        background-color: {DANGER_COLOR};
        color: white;
        border: 1px solid {DANGER_COLOR};
    }}
    
    /* Source citations */
    .source-citation {{
        background-color: #F5F5F5;

        padding: 10px 14px;
        border-radius: 8px;
        font-size: 13px;
        color: {NEUTRAL_DARK};
        margin: 6px 0;
        border-left: 4px solid {PRIMARY_COLOR};
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
        transform: translateY(-2px);
        color:white;
    }}
    
    /* Input field placeholder styling */
    .stTextInput > div > div > input::placeholder {{
        # color: {NEUTRAL_DARK};
        color:white;
        opacity: 0.6;
    }}
    
    /* Input styling */
    .stTextInput > div > div > input {{
        # background-color: #FFFFFF;
        # background-color: black;
        border: 2px solid #E5E5E5;
        # color: {NEUTRAL_DARK};
        color:white;
        border-radius: 8px;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1);

    }}
    
    /* Divider */
    .stDivider {{
        border-color: #E5E5E5;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: #F5F5F5;

        color: {NEUTRAL_DARK};

    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {{
        color: {NEUTRAL_DARK};
        border-bottom: 3px solid transparent;
    }}
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom-color: {PRIMARY_COLOR};
        color: {PRIMARY_COLOR};
        font-weight: 600;
    }}
    
    /* Info/Warning/Error boxes */
    
    
    .stInfo > div {{
        color: white;
    }}
    
    .stWarning {{
        background-color: {WARNING_COLOR};
        border-left-color: {WARNING_COLOR};
        color: white;
    }}
    
    .stError {{
        background-color: {DANGER_COLOR};
        border-left-color: {DANGER_COLOR};
        color: white;
    }}
    
    .stSuccess {{
        background-color: {SUCCESS_COLOR};
        border-left-color: {SUCCESS_COLOR};
        color: white;
    }}
    
    /* Spinner styling */
    .stSpinner {{
        color: {PRIMARY_COLOR};
  
    }}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {{
        color: {PRIMARY_COLOR}
    }}
    </style>
""", unsafe_allow_html=True)

# --- API Configuration ---
API_BASE_URL = "https://ai-engineering-projects-production.up.railway.app"
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
        st.error("❌ Cannot connect to the backend at https://ai-engineering-projects-production.up.railway.app . Start uvicorn with: `uvicorn app.main:app --reload`")
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

# --- Top Navigation Tabs ---
tab1, tab2 = st.tabs(["💬 Chat", "📊 Customer Risk"])

# --- Tab 1: Chat Interface ---
with tab1:
    st.markdown("### Ask BrightByte")
    st.markdown("Ask questions about policies, shipping, returns, warranties, and more.")

    # Chat history display
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            confidence_class = "" if message.get("confident", True) else " confidence-low"
            confidence_badge = "🟢 Confident" if message.get("confident", True) else "🟡 Low Confidence"
            
            st.markdown(f'<div class="bot-message{confidence_class}"><strong>Assistant:</strong>\n{message["content"]}<br><small>{confidence_badge}</small></div>', unsafe_allow_html=True)
            
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
            label_visibility="collapsed",
            key="chat_input"
        )

    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary", key="send_button")

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

# --- Tab 2: Customer Risk Dashboard ---
with tab2:
    st.markdown("### Customer Risk Dashboard")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown('<div class="refresh-button">', unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True):
    # your code
            st.markdown('</div>', unsafe_allow_html=True)
            with st.spinner("Loading customer data..."):
                customers = fetch_all_customers()
                if customers:
                    st.session_state.churn_data = score_all_customers_for_churn(customers)
                    st.markdown(f"""
<div style="
    background-color:66BB6A;
    border-left:5px solid #16A34A;
    padding:14px 16px;
    border-radius:10px;
    color:#111827;
    font-weight:600;
    font-size:16px;
">
    ✅ Scored {len(customers)} customers
</div>
""", unsafe_allow_html=True)
                    
                else:
                    st.error("Failed to fetch customer data")
    
    st.divider()
    
    if st.session_state.churn_data is not None and not st.session_state.churn_data.empty:
        df = st.session_state.churn_data
        
        # Summary metrics
        high_risk_count = len(df[df["churn_risk"] >= 0.7])
        medium_risk_count = len(df[(df["churn_risk"] >= 0.5) & (df["churn_risk"] < 0.7)])
        low_risk_count = len(df[df["churn_risk"] < 0.5])
        
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.metric("🔴 High Risk", high_risk_count)
        # with col2:
        #     st.metric("🟡 Medium Risk", medium_risk_count)
        # with col3:
        #     st.metric("🟢 Low Risk", low_risk_count)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
    <div style="
        background:#FEE2E2;
        padding:20px;
        border-radius:12px;
        border-left:5px solid #DC2626;
    ">
        <h4 style="margin:0;color:#DC2626;">🔴 High Risk</h4>
        <h2 style="margin:8px 0;color:#111827;">{high_risk_count}</h2>
    </div>
    """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
    <div style="
        background:#FEF3C7;
        padding:20px;
        border-radius:12px;
        border-left:5px solid #FACC15;
    ">
        <h4 style="margin:0;color:#92400E;">🟡 Medium Risk</h4>
        <h2 style="margin:8px 0;color:#111827;">{medium_risk_count}</h2>
    </div>
    """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
    <div style="
        background:#DCFCE7;
        padding:20px;
        border-radius:12px;
        border-left:5px solid #16A34A;
    ">
        <h4 style="margin:0;color:#166534;">🟢 Low Risk</h4>
        <h2 style="margin:8px 0;color:#111827;">{low_risk_count}</h2>
    </div>
    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Tabs for different risk levels
        risk_tab1, risk_tab2, risk_tab3 = st.tabs(["🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"])
        
        with risk_tab1:
            high_risk_df = df[df["churn_risk"] >= 0.7]
            if not high_risk_df.empty:
                for idx, row in high_risk_df.head(10).iterrows():
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
        
        with risk_tab2:
            medium_risk_df = df[(df["churn_risk"] >= 0.5) & (df["churn_risk"] < 0.7)]
            if not medium_risk_df.empty:
                for idx, row in medium_risk_df.head(10).iterrows():
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
        
        with risk_tab3:
            low_risk_df = df[df["churn_risk"] < 0.5]
            if not low_risk_df.empty:
                st.markdown(f"""
<div style="
    background-color:66BB6A;
    border-left:5px solid #16A34A;
    padding:14px 16px;
    border-radius:10px;
    color:#111827;
    font-weight:600;
    font-size:16px;
">
    ✅ {len(low_risk_df)} customers are stable (low risk)
</div>
""", unsafe_allow_html=True)
                st.dataframe(low_risk_df.head(10), use_container_width=True)
            else:
                st.info("No low-risk customers data available")
    
    else:
        st.markdown(f"""
<div style="
    background-color:66BB6A;
    border-left:5px solid black;
    padding:14px 16px;
    border-radius:10px;
    color:#111827;
    font-weight:600;
    font-size:16px;
">
    st.info("Click 'Refresh Data' to load and score all customers")
</div>
""", unsafe_allow_html=True)
        

# --- Footer ---
st.divider()
st.markdown(f"""
<div style="text-align: center; color: gray; font-size: 12px; padding: 20px 0; opacity: 0.75;">
    <p style="margin: 8px 0; font-weight: 500;">BrightByte AI Copilot v1.0</p>
    <p style="margin: 4px 0; font-size: 11px;">Enterprise RAG • Predictive Analytics • Intelligent Automation</p>
    <p style="margin: 8px 0; font-size: 10px; letter-spacing: 0.5px;">
        <strong>Powered by:</strong> FastAPI • ChromaDB • Google Gemini • n8n Workflows
    </p>
</div>
""", unsafe_allow_html=True)
