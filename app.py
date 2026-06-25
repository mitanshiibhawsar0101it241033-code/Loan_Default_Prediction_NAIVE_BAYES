import streamlit as st
import joblib
import numpy as np
import warnings
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NEXUS CREDIT · Risk Oracle",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    return joblib.load("nb_model_cleaned.pkl")

model = load_model()

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Bebas+Neue&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #0a0a0f !important;
    color: #e8e4dc !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 17px !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #07070c !important;
    border-right: 1px solid #1a1a2e !important;
    width: 260px !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label span {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    color: #8888aa !important;
}

/* Sidebar nav items */
div[data-testid="stRadio"] > label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #8888aa !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    padding: 10px 14px !important;
    border: 1px solid #1a1a2e !important;
    margin-bottom: 6px !important;
    border-radius: 0 !important;
    width: 100%;
    display: block;
    cursor: pointer;
    transition: all 0.2s;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    border-color: #b87333 !important;
    background: #12121e !important;
}

/* ── INPUTS ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] input {
    background: #0d0d18 !important;
    border: 1px solid #252540 !important;
    border-radius: 0 !important;
    color: #e8e4dc !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    padding: 10px 14px !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus {
    border-color: #b87333 !important;
    box-shadow: 0 0 0 2px #b8733322 !important;
}

/* Labels */
label, .stSlider label, .stNumberInput label, .stSelectbox label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    color: #8888aa !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

/* Sliders */
.stSlider > div > div > div { background: #252540 !important; }
.stSlider > div > div > div > div { background: #b87333 !important; }

/* Button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #b87333, #8b5e28) !important;
    border: none !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.4rem !important;
    letter-spacing: 0.2em !important;
    padding: 14px 40px !important;
    border-radius: 0 !important;
    width: 100% !important;
    cursor: pointer;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: #1a1a2e !important; }

/* Plotly chart container */
.js-plotly-plot { border: 1px solid #1a1a2e; }

/* Section headers */
.page-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    letter-spacing: 0.12em;
    color: #e8e4dc;
    line-height: 1;
    margin-bottom: 4px;
}
.page-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    color: #b87333;
    text-transform: uppercase;
    margin-bottom: 32px;
}
.section-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    color: #b87333;
    text-transform: uppercase;
    border-left: 2px solid #b87333;
    padding-left: 10px;
    margin-bottom: 16px;
    margin-top: 28px;
}
.card {
    background: #0d0d18;
    border: 1px solid #1a1a2e;
    padding: 22px 24px;
    margin-bottom: 14px;
}
.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    color: #8888aa;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.card-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 0.06em;
    color: #e8e4dc;
    line-height: 1;
}
.card-unit {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #8888aa;
    margin-left: 4px;
}

/* Result banner */
.result-safe {
    background: linear-gradient(135deg, #0a1a0f 0%, #0d1a12 100%);
    border: 1px solid #1e4d2a;
    border-left: 4px solid #3d9b5a;
    padding: 28px 32px;
    margin: 20px 0;
}
.result-risk {
    background: linear-gradient(135deg, #1a0a0a 0%, #1a0d0d 100%);
    border: 1px solid #4d1e1e;
    border-left: 4px solid #9b3d3d;
    padding: 28px 32px;
    margin: 20px 0;
}
.result-verdict {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 0.1em;
    line-height: 1;
}
.result-verdict.safe { color: #5ecb7e; }
.result-verdict.risk { color: #cb5e5e; }
.result-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: #8888aa;
    margin-top: 8px;
    line-height: 1.7;
}
.prob-strip {
    margin-top: 18px;
    display: flex;
    gap: 24px;
    align-items: center;
}
.prob-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.prob-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    line-height: 1;
}
.prob-pct.safe { color: #5ecb7e; }
.prob-pct.risk { color: #cb5e5e; }
.prob-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    color: #8888aa;
    text-transform: uppercase;
}
.divbar {
    width: 1px;
    height: 50px;
    background: #252540;
}

/* Feature importance row */
.feat-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.feat-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #8888aa;
    width: 180px;
    flex-shrink: 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.feat-bar-bg {
    flex: 1;
    height: 6px;
    background: #1a1a2e;
}
.feat-bar-fill {
    height: 100%;
}
.feat-val {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #e8e4dc;
    width: 44px;
    text-align: right;
}

/* Insight box */
.insight-box {
    background: #0d0d18;
    border: 1px solid #1a1a2e;
    border-top: 2px solid #b87333;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.insight-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    color: #b87333;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.insight-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    color: #c0bbb0;
    line-height: 1.65;
}

/* Sidebar logo */
.sidebar-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.9rem;
    letter-spacing: 0.18em;
    color: #e8e4dc;
    margin-bottom: 2px;
}
.sidebar-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    color: #b87333;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* About page */
.about-big {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.5rem;
    letter-spacing: 0.06em;
    color: #1a1a2e;
    line-height: 1;
    position: absolute;
    top: 0; right: 0;
    pointer-events: none;
    user-select: none;
}
.metric-big {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 0.06em;
    color: #b87333;
    line-height: 1;
}
.metric-desc {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #8888aa;
    text-transform: uppercase;
}

/* Number input stepper */
div[data-testid="stNumberInput"] button {
    background: #1a1a2e !important;
    border: none !important;
    color: #8888aa !important;
    border-radius: 0 !important;
}

/* Selectbox arrow */
div[data-testid="stSelectbox"] svg { color: #8888aa !important; }

/* Streamlit column gap fix */
[data-testid="column"] { padding: 0 8px !important; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">⬡ NEXUS</div>
    <div class="sidebar-tagline">Credit Risk Oracle</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "NAVIGATE",
        ["⬡  Overview", "◈  Applicant Input", "◉  Risk Assessment", "◎  Model Insights"],
        index=0,
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#3a3a55;text-transform:uppercase;letter-spacing:0.12em;line-height:2;'>
    Model: Gaussian NB<br>
    Features: 22<br>
    Classes: Default / Safe<br>
    Base rate: 21.5%
    </div>
    """, unsafe_allow_html=True)

# ─── HELPER: Plotly dark theme ─────────────────────────────────────────────────
PLOT_BG = "#0a0a0f"
PAPER_BG = "#0a0a0f"
GRID = "#1a1a2e"
TEXT_C = "#8888aa"
COPPER = "#b87333"
SAFE_C = "#3d9b5a"
RISK_C = "#9b3d3d"

def dark_layout(fig, title="", height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Space Mono, monospace", color=TEXT_C, size=11),
        title=dict(text=title, font=dict(family="Bebas Neue,sans-serif", size=22, color="#e8e4dc"), x=0, pad=dict(l=4)),
        margin=dict(l=16, r=16, t=48 if title else 16, b=16),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, color=TEXT_C),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, color=TEXT_C),
        legend=dict(bgcolor="#0d0d18", bordercolor=GRID, borderwidth=1,
                    font=dict(family="Space Mono,monospace", size=10, color=TEXT_C)),
    )
    return fig

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
if "⬡  Overview" in page:
    st.markdown("""
    <div class="page-title">CREDIT RISK<br>ORACLE</div>
    <div class="page-subtitle">⬡ NEXUS · Gaussian Naïve Bayes · Loan Default Prediction</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Model", "Gaussian NB", ""),
        ("Features", "22", "vars"),
        ("Base Default Rate", "21.5", "%"),
        ("Target Classes", "2", "binary"),
    ]
    for col, (lbl, val, unit) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">{lbl}</div>
                <div class="card-value">{val}<span class="card-unit">{unit}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<div class="section-tag">Class Distribution · Training Prior</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["No Default (Safe)", "Default (Risk)"],
            values=[78.5, 21.5],
            hole=0.62,
            marker=dict(colors=[SAFE_C, RISK_C], line=dict(color=PLOT_BG, width=3)),
            textfont=dict(family="Space Mono, monospace", size=11, color="#e8e4dc"),
            textinfo="percent+label",
        ))
        fig_pie.add_annotation(
            text="<b>78.5%</b><br>SAFE",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Bebas Neue, sans-serif", size=20, color=SAFE_C),
            align="center"
        )
        dark_layout(fig_pie, height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-tag">Key Feature Means · Safe vs Default</div>', unsafe_allow_html=True)
        feat_labels = ["Income", "Credit Score", "Loan Amount", "Interest Rate", "Age"]
        safe_vals  = [64586, 628, 7339, 9.81, 23.6]
        risk_vals  = [46708, 625, 7968, 12.03, 23.4]
        # normalise to 0-100 for display
        max_v = [max(s,r)*1.1 for s,r in zip(safe_vals, risk_vals)]
        safe_n = [s/m*100 for s,m in zip(safe_vals, max_v)]
        risk_n = [r/m*100 for r,m in zip(risk_vals, max_v)]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Safe", y=feat_labels, x=safe_n, orientation='h',
            marker_color=SAFE_C, opacity=0.85,
        ))
        fig_bar.add_trace(go.Bar(
            name="Default", y=feat_labels, x=risk_n, orientation='h',
            marker_color=RISK_C, opacity=0.85,
        ))
        fig_bar.update_layout(barmode='group', height=300)
        fig_bar.update_xaxes(title_text="Relative Scale (%)", title_font=dict(size=10))
        dark_layout(fig_bar, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-tag">How It Works</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    steps = [
        ("01 · INPUT", "Enter applicant demographics, loan parameters, and credit history across the input page."),
        ("02 · CLASSIFY", "The Gaussian Naïve Bayes model computes posterior probabilities across 22 engineered features."),
        ("03 · ASSESS", "View the verdict, probability breakdown, and contextual risk insights on the assessment page."),
    ]
    for col, (title, text) in zip([s1,s2,s3], steps):
        with col:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 2 — APPLICANT INPUT
# ════════════════════════════════════════════════════════════════════
elif "◈  Applicant Input" in page:
    st.markdown("""
    <div class="page-title">APPLICANT<br>PROFILE</div>
    <div class="page-subtitle">◈ Fill all fields · Results appear on the Assessment page</div>
    """, unsafe_allow_html=True)

    # ── Section A: Personal ────────────────────────────────────────
    st.markdown('<div class="section-tag">A · Personal Details</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.session_state["person_age"] = st.number_input(
            "Age (years)", 18, 100, st.session_state.get("person_age", 30), 1)
    with a2:
        st.session_state["person_income"] = st.number_input(
            "Annual Income (₹)", 0, 10_000_000, st.session_state.get("person_income", 500000), 10000)
    with a3:
        st.session_state["person_emp_exp"] = st.number_input(
            "Employment Experience (yrs)", 0, 50, st.session_state.get("person_emp_exp", 5), 1)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.session_state["person_gender"] = st.selectbox(
            "Gender", ["Female", "Male"],
            index=["Female","Male"].index(st.session_state.get("person_gender","Female")))
    with b2:
        edu_opts = ["High School", "Bachelor", "Master", "Doctorate"]
        st.session_state["person_education"] = st.selectbox(
            "Education", edu_opts,
            index=edu_opts.index(st.session_state.get("person_education","Bachelor")))
    with b3:
        home_opts = ["RENT", "OWN", "MORTGAGE", "OTHER"]
        st.session_state["person_home_ownership"] = st.selectbox(
            "Home Ownership", home_opts,
            index=home_opts.index(st.session_state.get("person_home_ownership","RENT")))

    st.markdown("---")
    # ── Section B: Credit ─────────────────────────────────────────
    st.markdown('<div class="section-tag">B · Credit Standing</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["credit_score"] = st.number_input(
            "Credit Score", 300, 850, st.session_state.get("credit_score", 650), 1)
    with c2:
        st.session_state["cb_cred_hist"] = st.number_input(
            "Credit History Length (yrs)", 0, 30, st.session_state.get("cb_cred_hist", 5), 1)
    with c3:
        prev_opts = ["No", "Yes"]
        st.session_state["prev_default"] = st.selectbox(
            "Previous Loan Default on File", prev_opts,
            index=prev_opts.index(st.session_state.get("prev_default","No")))

    # Credit score visual
    cs = st.session_state["credit_score"]
    if cs < 580: cs_label, cs_color = "POOR", "#9b3d3d"
    elif cs < 670: cs_label, cs_color = "FAIR", "#9b6a3d"
    elif cs < 740: cs_label, cs_color = "GOOD", "#9b8a3d"
    elif cs < 800: cs_label, cs_color = "VERY GOOD", "#5a8c3d"
    else: cs_label, cs_color = "EXCEPTIONAL", "#3d9b5a"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cs,
        gauge=dict(
            axis=dict(range=[300,850], tickcolor=TEXT_C, tickfont=dict(size=10)),
            bar=dict(color=cs_color, thickness=0.25),
            bgcolor=PLOT_BG,
            borderwidth=0,
            steps=[
                dict(range=[300,580], color="#1a1014"),
                dict(range=[580,670], color="#1a140e"),
                dict(range=[670,740], color="#14180a"),
                dict(range=[740,800], color="#0e1810"),
                dict(range=[800,850], color="#0a1810"),
            ],
            threshold=dict(line=dict(color=COPPER, width=3), thickness=0.8, value=cs)
        ),
        number=dict(font=dict(family="Bebas Neue,sans-serif", size=42, color=cs_color),
                    suffix=""),
        title=dict(text=f"<b>{cs_label}</b>", font=dict(family="Space Mono,monospace", size=13, color=cs_color)),
        domain=dict(x=[0,1], y=[0,1])
    ))
    dark_layout(fig_gauge, "Credit Score Band", height=240)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    # ── Section C: Loan ───────────────────────────────────────────
    st.markdown('<div class="section-tag">C · Loan Parameters</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.session_state["loan_amnt"] = st.number_input(
            "Loan Amount (₹)", 500, 5_000_000, st.session_state.get("loan_amnt", 200000), 5000)
    with d2:
        st.session_state["loan_int_rate"] = st.slider(
            "Interest Rate (%)", 1.0, 30.0, st.session_state.get("loan_int_rate", 11.0), 0.1)
    with d3:
        st.session_state["loan_percent_income"] = st.slider(
            "Loan-to-Income Ratio", 0.0, 1.0, st.session_state.get("loan_percent_income", 0.20), 0.01)

    intent_opts = ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"]
    st.session_state["loan_intent"] = st.selectbox(
        "Loan Purpose", intent_opts,
        index=intent_opts.index(st.session_state.get("loan_intent","PERSONAL")))

    # Loan breakdown donut
    inc = st.session_state["person_income"]
    amnt = st.session_state["loan_amnt"]
    remaining = max(0, inc - amnt)
    fig_donut = go.Figure(go.Pie(
        labels=["Loan Amount", "Remaining Income"],
        values=[amnt, remaining] if remaining > 0 else [amnt, 1],
        hole=0.65,
        marker=dict(colors=[COPPER, "#1a1a2e"], line=dict(color=PLOT_BG, width=3)),
        textinfo="percent",
        textfont=dict(family="Space Mono,monospace", size=10),
    ))
    fig_donut.add_annotation(
        text=f"<b>{amnt/inc*100:.0f}%</b><br>of income",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="Bebas Neue,sans-serif", size=18, color=COPPER),
        align="center"
    )
    dark_layout(fig_donut, "Loan vs Annual Income", height=260)
    st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.info("⬡  All fields saved — navigate to **◉ Risk Assessment** to run the prediction.")

# ════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK ASSESSMENT
# ════════════════════════════════════════════════════════════════════
elif "◉  Risk Assessment" in page:
    st.markdown("""
    <div class="page-title">RISK<br>ASSESSMENT</div>
    <div class="page-subtitle">◉ Gaussian NB · Real-time classification</div>
    """, unsafe_allow_html=True)

    required = ["person_age","person_income","person_emp_exp","person_gender",
                "person_education","person_home_ownership","credit_score",
                "cb_cred_hist","prev_default","loan_amnt","loan_int_rate",
                "loan_percent_income","loan_intent"]

    missing = [k for k in required if k not in st.session_state]

    if missing:
        st.warning("⬡  Please complete the **Applicant Input** page first before running assessment.")
    else:
        if st.button("⬡  RUN RISK ASSESSMENT"):
            s = st.session_state
            gender_male = 1 if s["person_gender"]=="Male" else 0
            edu_b = 1 if s["person_education"]=="Bachelor" else 0
            edu_d = 1 if s["person_education"]=="Doctorate" else 0
            edu_h = 1 if s["person_education"]=="High School" else 0
            edu_m = 1 if s["person_education"]=="Master" else 0
            home_other = 1 if s["person_home_ownership"]=="OTHER" else 0
            home_own   = 1 if s["person_home_ownership"]=="OWN" else 0
            home_rent  = 1 if s["person_home_ownership"]=="RENT" else 0
            intent_edu  = 1 if s["loan_intent"]=="EDUCATION" else 0
            intent_home = 1 if s["loan_intent"]=="HOMEIMPROVEMENT" else 0
            intent_med  = 1 if s["loan_intent"]=="MEDICAL" else 0
            intent_per  = 1 if s["loan_intent"]=="PERSONAL" else 0
            intent_ven  = 1 if s["loan_intent"]=="VENTURE" else 0
            prev_def    = 1 if s["prev_default"]=="Yes" else 0

            X = np.array([[
                s["person_age"], s["person_income"], s["person_emp_exp"],
                s["loan_amnt"], s["loan_int_rate"], s["loan_percent_income"],
                s["cb_cred_hist"], s["credit_score"],
                gender_male,
                edu_b, edu_d, edu_h, edu_m,
                home_other, home_own, home_rent,
                intent_edu, intent_home, intent_med, intent_per, intent_ven,
                prev_def
            ]])

            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            p_safe = proba[0]
            p_risk = proba[1]

            st.session_state["_pred"] = pred
            st.session_state["_p_safe"] = float(p_safe)
            st.session_state["_p_risk"] = float(p_risk)
            st.session_state["_X"] = X

    if "_pred" in st.session_state:
        pred   = st.session_state["_pred"]
        p_safe = st.session_state["_p_safe"]
        p_risk = st.session_state["_p_risk"]
        X      = st.session_state["_X"]
        s      = st.session_state

        is_safe = pred == 0
        verdict_class = "safe" if is_safe else "risk"
        verdict_word  = "LOW RISK · APPROVE" if is_safe else "HIGH RISK · REVIEW"
        verdict_note  = (
            "Applicant profile aligns with historically safe borrowers. Credit criteria met."
            if is_safe else
            "Elevated default probability. Recommend collateral review or co-signer before approval."
        )

        st.markdown(f"""
        <div class="result-{'safe' if is_safe else 'risk'}">
            <div class="result-verdict {verdict_class}">{verdict_word}</div>
            <div class="result-meta">{verdict_note}</div>
            <div class="prob-strip">
                <div class="prob-item">
                    <div class="prob-pct safe">{p_safe:.1%}</div>
                    <div class="prob-lbl">Safe Probability</div>
                </div>
                <div class="divbar"></div>
                <div class="prob-item">
                    <div class="prob-pct risk">{p_risk:.1%}</div>
                    <div class="prob-lbl">Default Probability</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts ─────────────────────────────────────────────────
        r1, r2 = st.columns(2)

        with r1:
            st.markdown('<div class="section-tag">Probability Breakdown</div>', unsafe_allow_html=True)
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                x=["Safe (No Default)", "Default Risk"],
                y=[p_safe*100, p_risk*100],
                marker_color=[SAFE_C, RISK_C],
                text=[f"{p_safe:.1%}", f"{p_risk:.1%}"],
                textposition="outside",
                textfont=dict(family="Bebas Neue,sans-serif", size=18, color="#e8e4dc"),
                width=0.45,
            ))
            fig_prob.update_yaxes(range=[0, 110], title_text="Probability (%)")
            dark_layout(fig_prob, height=300)
            st.plotly_chart(fig_prob, use_container_width=True)

        with r2:
            st.markdown('<div class="section-tag">Risk Confidence Meter</div>', unsafe_allow_html=True)
            risk_pct = p_risk * 100
            needle_color = RISK_C if risk_pct > 50 else (COPPER if risk_pct > 25 else SAFE_C)
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                number=dict(suffix="%", font=dict(family="Bebas Neue,sans-serif", size=40, color=needle_color)),
                delta=dict(reference=21.5, increasing=dict(color=RISK_C), decreasing=dict(color=SAFE_C),
                           font=dict(family="Space Mono,monospace", size=11)),
                gauge=dict(
                    axis=dict(range=[0,100], tickfont=dict(size=9)),
                    bar=dict(color=needle_color, thickness=0.25),
                    bgcolor=PLOT_BG, borderwidth=0,
                    steps=[
                        dict(range=[0,25], color="#0a1810"),
                        dict(range=[25,50], color="#141810"),
                        dict(range=[50,75], color="#181408"),
                        dict(range=[75,100], color="#1a0808"),
                    ],
                    threshold=dict(line=dict(color=COPPER, width=2), thickness=0.75, value=21.5)
                ),
                title=dict(text="Default Risk %<br><sub>▲ vs 21.5% base rate</sub>",
                           font=dict(family="Space Mono,monospace", size=11, color=TEXT_C)),
                domain=dict(x=[0,1], y=[0,1])
            ))
            dark_layout(fig_g2, height=300)
            st.plotly_chart(fig_g2, use_container_width=True)

        # ── Applicant snapshot ─────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-tag">Applicant Snapshot</div>', unsafe_allow_html=True)
        dti = s["loan_amnt"] / s["person_income"] if s["person_income"] > 0 else 0
        r = s["loan_int_rate"] / 100 / 12
        n = 36
        emi = s["loan_amnt"] * r / (1-(1+r)**-n) if r > 0 else s["loan_amnt"]/n

        snap_cols = st.columns(5)
        snaps = [
            ("Age", f"{s['person_age']}", "yrs"),
            ("Annual Income", f"₹{s['person_income']:,.0f}", ""),
            ("Credit Score", f"{s['credit_score']}", "pts"),
            ("DTI Ratio", f"{dti:.2f}", "×"),
            ("Est. EMI (3yr)", f"₹{emi:,.0f}", "/mo"),
        ]
        for col, (lbl, val, unit) in zip(snap_cols, snaps):
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">{lbl}</div>
                    <div class="card-value" style="font-size:1.7rem;">{val}
                        <span class="card-unit">{unit}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Radar chart: applicant vs avg safe/risk borrower ──────
        st.markdown("---")
        st.markdown('<div class="section-tag">Applicant vs Population · Radar</div>', unsafe_allow_html=True)

        categories = ["Income (norm)", "Credit Score (norm)", "Age (norm)",
                      "Loan Amt (norm)", "Interest Rate (norm)", "Emp Exp (norm)"]
        # Normalize applicant vs class means (max scale)
        scales = [100000, 850, 70, 50000, 30, 30]
        applicant_vals = [
            s["person_income"]/scales[0]*100,
            s["credit_score"]/scales[1]*100,
            s["person_age"]/scales[2]*100,
            s["loan_amnt"]/scales[3]*100,
            s["loan_int_rate"]/scales[4]*100,
            s["person_emp_exp"]/scales[5]*100,
        ]
        safe_vals_r  = [64586/100000*100, 628/850*100, 23.6/70*100, 7339/50000*100, 9.81/30*100, 1.63/30*100]
        risk_vals_r  = [46708/100000*100, 625/850*100, 23.4/70*100, 7968/50000*100, 12.03/30*100, 1.47/30*100]

        cats_closed = categories + [categories[0]]
        app_closed  = applicant_vals + [applicant_vals[0]]
        safe_closed = safe_vals_r + [safe_vals_r[0]]
        risk_closed = risk_vals_r + [risk_vals_r[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=safe_closed, theta=cats_closed, fill='toself',
            name="Safe Avg", line=dict(color=SAFE_C, width=1.5),
            fillcolor=f"{SAFE_C}18", opacity=0.8))
        fig_radar.add_trace(go.Scatterpolar(r=risk_closed, theta=cats_closed, fill='toself',
            name="Default Avg", line=dict(color=RISK_C, width=1.5),
            fillcolor=f"{RISK_C}18", opacity=0.8))
        fig_radar.add_trace(go.Scatterpolar(r=app_closed, theta=cats_closed, fill='toself',
            name="This Applicant", line=dict(color=COPPER, width=2.5),
            fillcolor=f"{COPPER}25", opacity=0.9))
        fig_radar.update_layout(
            polar=dict(
                bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID,
                                tickfont=dict(size=9, color=TEXT_C), linecolor=GRID),
                angularaxis=dict(gridcolor=GRID, linecolor=GRID,
                                 tickfont=dict(family="Space Mono,monospace", size=10, color=TEXT_C))
            ),
            height=400,
            paper_bgcolor=PAPER_BG,
            font=dict(family="Space Mono,monospace", color=TEXT_C, size=11),
            legend=dict(bgcolor="#0d0d18", bordercolor=GRID, borderwidth=1,
                        font=dict(family="Space Mono,monospace", size=10, color=TEXT_C)),
            margin=dict(l=50, r=50, t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:80px 40px;'>
            <div style='font-family:Bebas Neue,sans-serif;font-size:6rem;color:#1a1a2e;line-height:1;'>◉</div>
            <div style='font-family:Space Mono,monospace;font-size:0.7rem;letter-spacing:0.2em;color:#3a3a55;
                        text-transform:uppercase;margin-top:16px;'>
                Complete applicant input first,<br>then return here to run the assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════
elif "◎  Model Insights" in page:
    st.markdown("""
    <div class="page-title">MODEL<br>INSIGHTS</div>
    <div class="page-subtitle">◎ Feature analysis · Class separation · GaussianNB internals</div>
    """, unsafe_allow_html=True)

    # ── Class prior ────────────────────────────────────────────────
    st.markdown('<div class="section-tag">Class Priors</div>', unsafe_allow_html=True)
    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Prior · No Default (Safe)</div>
            <div class="card-value" style="color:#3d9b5a;">78.5<span class="card-unit">%</span></div>
        </div>""", unsafe_allow_html=True)
    with cp2:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Prior · Default (Risk)</div>
            <div class="card-value" style="color:#9b3d3d;">21.5<span class="card-unit">%</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Mean feature values per class ─────────────────────────────
    st.markdown('<div class="section-tag">Feature Means · Safe vs Default Class</div>', unsafe_allow_html=True)

    feature_names_clean = [
        "Age","Income","Emp Experience","Loan Amount","Interest Rate",
        "Loan/Income Ratio","Credit Hist Length","Credit Score",
        "Male","Edu: Bachelor","Edu: Doctorate","Edu: HighSchool","Edu: Master",
        "Home: Other","Home: Own","Home: Rent",
        "Intent: Education","Intent: HomeImprove","Intent: Medical",
        "Intent: Personal","Intent: Venture","Prev Default"
    ]
    safe_means = model.theta_[0]
    risk_means = model.theta_[1]

    # Top 10 continuous features for visibility
    top_idx = [0,1,2,3,4,5,6,7]
    names_t = [feature_names_clean[i] for i in top_idx]
    safe_t  = [safe_means[i] for i in top_idx]
    risk_t  = [risk_means[i] for i in top_idx]

    # Normalize each feature to 0-1
    max_t = [max(abs(s),abs(r))+1e-9 for s,r in zip(safe_t, risk_t)]
    safe_norm = [s/m*100 for s,m in zip(safe_t, max_t)]
    risk_norm = [r/m*100 for r,m in zip(risk_t, max_t)]

    fig_feat = go.Figure()
    fig_feat.add_trace(go.Bar(name="Safe (Class 0)", x=names_t, y=safe_norm,
                              marker_color=SAFE_C, opacity=0.85))
    fig_feat.add_trace(go.Bar(name="Default (Class 1)", x=names_t, y=risk_norm,
                              marker_color=RISK_C, opacity=0.85))
    fig_feat.update_layout(barmode="group")
    fig_feat.update_yaxes(title_text="Normalized Value (%)")
    dark_layout(fig_feat, "Continuous Feature Means by Class", height=340)
    st.plotly_chart(fig_feat, use_container_width=True)

    # ── Binary feature comparison ──────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-tag">Binary Feature Proportions · Safe vs Default</div>', unsafe_allow_html=True)

    bin_idx = list(range(8, 22))
    bin_names = [feature_names_clean[i] for i in bin_idx]
    bin_safe  = [safe_means[i]*100 for i in bin_idx]
    bin_risk  = [risk_means[i]*100 for i in bin_idx]

    fig_bin = go.Figure()
    fig_bin.add_trace(go.Bar(name="Safe", x=bin_names, y=bin_safe,
                             marker_color=SAFE_C, opacity=0.8))
    fig_bin.add_trace(go.Bar(name="Default", x=bin_names, y=bin_risk,
                             marker_color=RISK_C, opacity=0.8))
    fig_bin.update_layout(barmode="group")
    fig_bin.update_yaxes(title_text="Proportion in Class (%)")
    fig_bin.update_xaxes(tickangle=-38, tickfont=dict(size=9))
    dark_layout(fig_bin, "Binary Feature Rates by Class", height=340)
    st.plotly_chart(fig_bin, use_container_width=True)

    # ── Feature variance heatmap ───────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-tag">Feature Variance · Model Uncertainty per Class</div>', unsafe_allow_html=True)

    vars_safe = model.var_[0][:8]
    vars_risk = model.var_[1][:8]
    var_names  = feature_names_clean[:8]

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=var_names, y=np.log1p(vars_safe), mode='lines+markers',
        name="Safe Class Variance (log)", line=dict(color=SAFE_C, width=2),
        marker=dict(size=7, color=SAFE_C)
    ))
    fig_var.add_trace(go.Scatter(
        x=var_names, y=np.log1p(vars_risk), mode='lines+markers',
        name="Default Class Variance (log)", line=dict(color=RISK_C, width=2),
        marker=dict(size=7, color=RISK_C)
    ))
    fig_var.update_yaxes(title_text="log(1 + variance)")
    fig_var.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    dark_layout(fig_var, "Feature Variance by Class (log scale)", height=320)
    st.plotly_chart(fig_var, use_container_width=True)

    # ── Insight boxes ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-tag">Key Takeaways</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    insights = [
        ("Income Gap", "Safe borrowers average ~₹64,586 annual income vs ₹46,708 for defaulters — a 38% difference that strongly separates the classes."),
        ("Interest Rate Signal", "Default-class borrowers carry significantly higher interest rates (12.0% vs 9.8%), likely reflecting lenders' own risk pricing."),
        ("Previous Default Weight", "A prior default on file is the single strongest categorical signal for repeated default, dominating binary feature separation."),
    ]
    for col, (title, text) in zip([i1,i2,i3], insights):
        with col:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='border-top:1px solid #1a1a2e;padding:14px 0 0 0;margin-top:32px;
            display:flex;justify-content:space-between;align-items:center;'>
    <span style='font-family:Space Mono,monospace;font-size:0.56rem;color:#2a2a40;
                 letter-spacing:0.14em;text-transform:uppercase;'>
        ⬡ NEXUS CREDIT · Gaussian Naïve Bayes · 22 Features · Binary Classification
    </span>
    <span style='font-family:Space Mono,monospace;font-size:0.56rem;color:#2a2a40;
                 letter-spacing:0.12em;text-transform:uppercase;'>
        For decisioning support only · Not a substitute for credit officer review
    </span>
</div>
""", unsafe_allow_html=True)