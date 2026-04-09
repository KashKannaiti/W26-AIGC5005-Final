# ══════════════════════════════════════════════════════════════════════════════
# app.py — Diabetes 30-Day Readmission Risk Predictor
# Project : AIGC 5005 — AI Capstone Project Preparation
# Team    : Fadi Kash Kannaiti | Ogbeide Iria | Oguzhan Tekin | Sara Yenigun
# School  : Humber Polytechnic
#
# Run locally : streamlit run app.py
# Deploy free : https://share.streamlit.io  (connect your GitHub repo)
# ══════════════════════════════════════════════════════════════════════════════

import os
import json
import warnings
import numpy  as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title  = "DiabetesRisk AI",
    page_icon   = "🏥",
    layout      = "centered",
    initial_sidebar_state = "collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Custom CSS
# Clean medical aesthetic: navy + white + red accent, IBM Plex Mono headers
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Source+Serif+4:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', serif;
}

/* ── App background ── */
.stApp { background-color: #F7F8FA; }

/* ── Main header ── */
.app-header {
    background: linear-gradient(135deg, #0A2540 0%, #1a3a5c 100%);
    border-radius: 16px;
    padding: 32px 36px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(220,50,47,0.12);
}
.app-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    color: #FFFFFF;
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}
.app-header p {
    color: #90A8C3;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.app-header .badge {
    display: inline-block;
    background: rgba(220,50,47,0.85);
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

/* ── Section labels ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: #0A2540;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-left: 3px solid #DC322F;
    padding-left: 10px;
    margin: 24px 0 14px 0;
}

/* ── Risk result cards ── */
.risk-card {
    border-radius: 14px;
    padding: 24px 28px;
    margin: 20px 0;
    border-left: 5px solid;
}
.risk-high   { background:#FFF0F0; border-color:#DC322F; }
.risk-medium { background:#FFFBF0; border-color:#E8A020; }
.risk-low    { background:#F0FFF4; border-color:#2DA44E; }

.risk-card .risk-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0 0 8px 0;
}
.risk-high   .risk-title { color: #A61010; }
.risk-medium .risk-title { color: #8A5A00; }
.risk-low    .risk-title { color: #1A6E32; }

.risk-card .risk-body {
    font-size: 0.92rem;
    line-height: 1.6;
    margin: 0;
}
.risk-high   .risk-body { color: #5C1A1A; }
.risk-medium .risk-body { color: #5C3D00; }
.risk-low    .risk-body { color: #1A4A25; }

/* ── Metric tiles ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}
.metric-tile {
    flex: 1;
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-tile .m-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #0A2540;
}
.metric-tile .m-lbl {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 2px;
    letter-spacing: 0.5px;
}

/* ── Progress bar ── */
.prob-bar-wrap { margin: 14px 0 6px; }
.prob-bar-track {
    height: 10px;
    background: #E2E8F0;
    border-radius: 5px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 10px;
    border-radius: 5px;
    transition: width 0.6s ease;
}

/* ── Info box ── */
.info-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #1E40AF;
    line-height: 1.6;
    margin: 12px 0;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #E2E8F0;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > div { padding: 4px 0; }
.stSelectbox > div > div { border-radius: 8px; }
div[data-testid="stNumberInput"] input { border-radius: 8px; }
.stButton > button {
    background: #0A2540 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model & Metadata Loading
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = "/content"          # Colab path
LOCAL_DIR  = "."                 # Local / deployed path

def resolve_path(filename: str) -> str:
    """Return the correct path for model files whether running in Colab or locally."""
    colab_path = os.path.join(BASE_DIR, "models", filename)
    local_path = os.path.join(LOCAL_DIR, "models", filename)
    if os.path.exists(colab_path):
        return colab_path
    if os.path.exists(local_path):
        return local_path
    return local_path  # Will raise a clean error below if missing

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model_path    = resolve_path("best_model.pkl")
    features_path = resolve_path("feature_names.json")
    meta_path     = resolve_path("model_metadata.json")

    if not os.path.exists(model_path):
        st.error(
            f"❌  Model file not found at `{model_path}`.\n\n"
            "**Steps to fix:**\n"
            "1. Run Notebook 04 first to generate `best_model.pkl`\n"
            "2. Make sure the `/content/models/` folder exists in Colab, OR\n"
            "3. Place `best_model.pkl` and `feature_names.json` in a `models/` "
            "   folder next to `app.py`"
        )
        st.stop()

    model    = joblib.load(model_path)
    features = json.load(open(features_path)) if os.path.exists(features_path) else []
    meta     = json.load(open(meta_path))     if os.path.exists(meta_path)     else {}
    return model, features, meta

model, feature_names, meta = load_model()
MODEL_NAME = meta.get("best_model_name", "Best Model")
MODEL_F1   = meta.get("best_f1", 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(inputs: dict) -> pd.DataFrame:
    """
    Map the user-facing form inputs onto the model's full feature vector.
    All features not explicitly set are left at 0 (their default encoded value).
    """
    vec = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    # Direct numeric mappings
    direct = {
        "time_in_hospital":  inputs["time_in_hospital"],
        "num_lab_procedures":inputs["num_lab_procedures"],
        "num_procedures":    inputs["num_procedures"],
        "num_medications":   inputs["num_medications"],
        "number_outpatient": inputs["number_outpatient"],
        "number_emergency":  inputs["number_emergency"],
        "number_inpatient":  inputs["number_inpatient"],
        "number_diagnoses":  inputs["number_diagnoses"],
        "diag_1":            inputs["diag_1"],
        "diag_2":            inputs["diag_2"],
        "diag_3":            inputs["diag_3"],
        "max_glu_serum":     inputs["max_glu_serum"],
        "A1Cresult":         inputs["A1Cresult"],
        "insulin":           inputs["insulin_enc"],
        "change":            inputs["change_enc"],
        "diabetesMed":       inputs["diabetesMed_enc"],
        "gender":            inputs["gender_enc"],
    }
    for col, val in direct.items():
        if col in vec.columns:
            vec[col] = val

    # Age one-hot — sanitised column names match what Cell 2 produced
    age_col_map = {
        "[10-20)": "age_10_20_",  "[20-30)": "age_20_30_",
        "[30-40)": "age_30_40_",  "[40-50)": "age_40_50_",
        "[50-60)": "age_50_60_",  "[60-70)": "age_60_70_",
        "[70-80)": "age_70_80_",  "[80-90)": "age_80_90_",
        "[90-100)":"age_90_100_",
    }
    age_val = inputs.get("age", "[0-10)")
    for label, col_fragment in age_col_map.items():
        # Match any column that contains the fragment (handles minor naming variations)
        matched = [c for c in vec.columns if col_fragment.lower() in c.lower()]
        for c in matched:
            vec[c] = 1 if label == age_val else 0

    # Race one-hot
    race_map = {
        "Caucasian":      "race_Caucasian",
        "AfricanAmerican":"race_AfricanAmerican",
        "Hispanic":       "race_Hispanic",
        "Asian":          "race_Asian",
        "Other":          "race_Other",
    }
    race_val = inputs.get("race", "Caucasian")
    for label, col in race_map.items():
        matched = [c for c in vec.columns if col.lower() in c.lower()]
        for c in matched:
            vec[c] = 1 if label == race_val else 0

    return vec


def risk_label(prob: float) -> tuple:
    """Return (level_str, css_class, bar_color, action_text) for a given probability."""
    if prob >= 0.50:
        return (
            "HIGH RISK",
            "risk-high",
            "#DC322F",
            "Schedule follow-up within 7 days. Review medication adherence plan. "
            "Consider referral to a diabetes care coordinator. Ensure patient has "
            "contact details for their care team before discharge."
        )
    elif prob >= 0.30:
        return (
            "MODERATE RISK",
            "risk-medium",
            "#E8A020",
            "Follow-up call recommended within 14 days. Confirm patient understands "
            "discharge medications. Provide written instructions for warning signs "
            "that require urgent care."
        )
    else:
        return (
            "LOW RISK",
            "risk-low",
            "#2DA44E",
            "Standard discharge protocol is appropriate. Provide routine follow-up "
            "appointment within 30 days. Ensure patient has prescriptions filled "
            "before leaving."
        )


def make_gauge(prob: float, bar_color: str):
    """Draw a minimal horizontal gauge bar using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 0.55))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Track
    ax.barh(0, 1, height=0.55, color="#E2E8F0", left=0)
    # Fill
    ax.barh(0, prob, height=0.55, color=bar_color, left=0)
    # Threshold markers
    for thr, clr in [(0.30, "#E8A020"), (0.50, "#DC322F")]:
        ax.axvline(thr, color=clr, linewidth=1.4, linestyle="--", alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — App Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="app-header">
  <div class="badge">AIGC 5005 · HUMBER POLYTECHNIC</div>
  <h1>🏥 DiabetesRisk AI</h1>
  <p>30-Day Hospital Readmission Risk Predictor &nbsp;·&nbsp;
     Model: <strong style="color:#fff">{MODEL_NAME}</strong> &nbsp;·&nbsp;
     F1-score: <strong style="color:#fff">{MODEL_F1:.3f}</strong></p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
  Enter patient discharge information below. The model predicts the probability
  that this patient will be readmitted to hospital within 30 days.
  Results are for <strong>educational demonstration purposes only</strong> and
  are not intended for clinical decision-making.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Input Form
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Patient Demographics</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.selectbox(
        "Age group",
        ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
         "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"],
        index=6,
        help="Patient's age bracket at time of admission"
    )
with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
with col3:
    race = st.selectbox(
        "Race",
        ["Caucasian","AfricanAmerican","Hispanic","Asian","Other"],
        help="As recorded in the clinical dataset"
    )

st.markdown('<div class="section-label">Clinical Indicators</div>',
            unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4:
    time_in_hospital   = st.slider("Days in hospital",          1, 14, 4,
                                   help="Length of current hospital stay")
    num_medications    = st.slider("Number of medications",      1, 81, 15,
                                   help="Total distinct medications prescribed")
    number_inpatient   = st.slider("Prior inpatient visits (1 yr)", 0, 21, 1,
                                   help="Number of inpatient visits in the past year")
    number_emergency   = st.slider("Emergency visits (1 yr)",    0, 42, 0,
                                   help="Number of emergency visits in the past year")

with col5:
    number_diagnoses   = st.slider("Number of diagnoses",        1,  9, 6,
                                   help="Total diagnoses entered in the system")
    num_lab_procedures = st.slider("Lab procedures",             1, 132, 45,
                                   help="Number of lab tests performed")
    num_procedures     = st.slider("Other procedures",           0,  6, 1,
                                   help="Number of non-lab procedures performed")
    number_outpatient  = st.slider("Outpatient visits (1 yr)",   0, 42, 0,
                                   help="Number of outpatient visits in the past year")

st.markdown('<div class="section-label">Diagnosis & Lab Results</div>',
            unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    diag1_label = st.selectbox(
        "Primary diagnosis category",
        ["Other/External","Circulatory","Respiratory","Digestive",
         "Diabetes","Injury/Poisoning","Musculoskeletal","Genitourinary","Neoplasms"],
        index=1,
        help="Broad ICD-9 category of principal diagnosis"
    )
    diag2_label = st.selectbox(
        "Secondary diagnosis category",
        ["Other/External","Circulatory","Respiratory","Digestive",
         "Diabetes","Injury/Poisoning","Musculoskeletal","Genitourinary","Neoplasms"],
        index=4
    )
    diag3_label = st.selectbox(
        "Tertiary diagnosis category",
        ["Other/External","Circulatory","Respiratory","Digestive",
         "Diabetes","Injury/Poisoning","Musculoskeletal","Genitourinary","Neoplasms"],
        index=0
    )

with col7:
    max_glu_label = st.selectbox(
        "Max glucose serum test result",
        ["None / not measured", "Normal", ">200 mg/dL", ">300 mg/dL"],
        help="Result of maximum glucose serum test, if taken"
    )
    a1c_label = st.selectbox(
        "HbA1c (A1C) result",
        ["None / not measured", "Normal (<7%)", ">7%", ">8%"],
        help="Most recent HbA1c test result"
    )

st.markdown('<div class="section-label">Medications</div>',
            unsafe_allow_html=True)

col8, col9, col10 = st.columns(3)
with col8:
    insulin_label = st.selectbox(
        "Insulin",
        ["No", "Steady", "Up", "Down"],
        index=1,
        help="Was insulin prescribed, and was the dose changed?"
    )
with col9:
    diabetes_med = st.selectbox(
        "Diabetes medication prescribed?",
        ["Yes", "No"],
        help="Was any diabetes medication prescribed during this encounter?"
    )
with col10:
    med_changed = st.selectbox(
        "Medication change?",
        ["Yes (change made)", "No (no change)"],
        help="Was there a change in diabetic medications during the encounter?"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Encode Inputs & Predict
# ══════════════════════════════════════════════════════════════════════════════

# Encode categorical selections to numeric values
diag_map = {"Other/External":0,"Circulatory":1,"Respiratory":2,"Digestive":3,
            "Diabetes":4,"Injury/Poisoning":5,"Musculoskeletal":6,
            "Genitourinary":7,"Neoplasms":8}

glu_map  = {"None / not measured":0,"Normal":1,">200 mg/dL":2,">300 mg/dL":3}
a1c_map  = {"None / not measured":0,"Normal (<7%)":1,">7%":2,">8%":3}
ins_map  = {"No":0,"Steady":1,"Down":2,"Up":3}

inputs = {
    "age":              age,
    "race":             race,
    "gender_enc":       0 if gender == "Female" else 1,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures":   num_procedures,
    "num_medications":  num_medications,
    "number_outpatient":number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "diag_1":           diag_map[diag1_label],
    "diag_2":           diag_map[diag2_label],
    "diag_3":           diag_map[diag3_label],
    "max_glu_serum":    glu_map[max_glu_label],
    "A1Cresult":        a1c_map[a1c_label],
    "insulin_enc":      ins_map[insulin_label],
    "change_enc":       1 if "Yes" in med_changed else 0,
    "diabetesMed_enc":  1 if diabetes_med == "Yes" else 0,
}

st.markdown("---")
predict_clicked = st.button("⚕  Predict Readmission Risk", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Results Display
# ══════════════════════════════════════════════════════════════════════════════
if predict_clicked:
    with st.spinner("Analysing patient data…"):
        X_input = build_feature_vector(inputs)
        prob    = float(model.predict_proba(X_input)[0][1])
        pct     = round(prob * 100, 1)
        level, css_class, bar_color, action = risk_label(prob)

    # ── Risk card ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="risk-card {css_class}">
      <p class="risk-title">{level} &nbsp;—&nbsp; {pct:.1f}% probability of readmission within 30 days</p>
      <p class="risk-body"><strong>Recommended action:</strong> {action}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge bar ─────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.82rem;color:#64748B;margin:6px 0 4px;font-family:IBM Plex Mono,monospace'>"
        "RISK SCORE &nbsp; │ &nbsp; ── = 30% threshold &nbsp; ── = 50% threshold</p>",
        unsafe_allow_html=True)
    st.pyplot(make_gauge(prob, bar_color), use_container_width=True)
    plt.close("all")

    # ── Metric tiles ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile">
        <div class="m-val">{pct:.1f}%</div>
        <div class="m-lbl">Readmission probability</div>
      </div>
      <div class="metric-tile">
        <div class="m-val">{round((1-prob)*100,1)}%</div>
        <div class="m-lbl">No readmission probability</div>
      </div>
      <div class="metric-tile">
        <div class="m-val">{MODEL_F1:.3f}</div>
        <div class="m-lbl">Model F1-score</div>
      </div>
      <div class="metric-tile">
        <div class="m-val">{number_inpatient}</div>
        <div class="m-lbl">Prior inpatient visits</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top contributing factors ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Key risk factors for this patient</div>',
                unsafe_allow_html=True)

    # Build a ranked list of the patient's most clinically meaningful values
    factor_scores = {
        f"Prior inpatient visits: {number_inpatient}":  number_inpatient * 3,
        f"Days in hospital: {time_in_hospital}":         time_in_hospital * 1.5,
        f"Number of diagnoses: {number_diagnoses}":      number_diagnoses * 1.2,
        f"Medications prescribed: {num_medications}":    num_medications  * 0.8,
        f"Emergency visits: {number_emergency}":         number_emergency * 2,
        f"Primary diagnosis: {diag1_label}":             6 if diag1_label in
                                                         ["Circulatory","Diabetes"] else 2,
        f"Insulin: {insulin_label}":                     4 if insulin_label in
                                                         ["Up","Down"] else 1,
        f"HbA1c: {a1c_label}":                          3 if ">8" in a1c_label else 1,
    }
    sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
    top_factors    = sorted_factors[:5]
    max_score      = max(v for _, v in top_factors) or 1

    fig2, ax2 = plt.subplots(figsize=(6, 2.6))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor("none")

    labels_plot = [f[0] for f in top_factors][::-1]
    values_plot = [f[1] for f in top_factors][::-1]
    bar_colors_plot = [bar_color if v == max(values_plot) else "#90A8C3"
                       for v in values_plot]

    bars2 = ax2.barh(labels_plot, values_plot, color=bar_colors_plot,
                     edgecolor="white", height=0.55)
    ax2.set_xlim(0, max_score * 1.25)
    ax2.set_xlabel("Relative contribution score", fontsize=9, color="#64748B")
    ax2.tick_params(axis="y", labelsize=9, colors="#0A2540")
    ax2.tick_params(axis="x", labelsize=8, colors="#94A3B8")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.xaxis.grid(True, alpha=0.25)
    ax2.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig2, use_container_width=True)
    plt.close("all")

    # ── Clinical note ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
      <strong>About this prediction:</strong> The model was trained on the UCI Diabetes
      130-US Hospitals dataset (101,766 patient encounters from 130 US hospitals,
      1999–2008). It does not incorporate social determinants of health, real-time
      lab values, or physician assessments. Always combine model output with clinical
      judgment.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Sidebar: Model Info & How to Use
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### About this app")
    st.markdown(f"""
**Model:** {MODEL_NAME}

**F1-score:** {MODEL_F1:.4f}

**Dataset:** UCI Diabetes 130-US Hospitals  
101,766 patient encounters  
130 US hospitals, 1999–2008

**Team:**
- Fadi Kash Kannaiti
- Ogbeide Iria
- Oguzhan Tekin
- Sara Yenigun

**Course:** AIGC 5005  
**Institution:** Humber Polytechnic
    """)

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
1. Fill in all patient fields in the main panel
2. Click **Predict Readmission Risk**
3. Review the risk level and recommended action
4. Use clinical judgment alongside the prediction

**Risk thresholds:**
- 🔴 ≥ 50% → High Risk
- 🟡 30–49% → Moderate Risk
- 🟢 < 30% → Low Risk
    """)

    st.markdown("---")
    st.markdown("### Disclaimer")
    st.caption(
        "This application is for educational and demonstration purposes only. "
        "It is not a certified medical device and must not be used for clinical "
        "diagnosis or treatment decisions."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
  AIGC 5005 · Humber Polytechnic · Winter 2026 &nbsp;|&nbsp;
  Fadi Kash Kannaiti &nbsp;·&nbsp; Ogbeide Iria &nbsp;·&nbsp;
  Oguzhan Tekin &nbsp;·&nbsp; Sara Yenigun
</div>
""", unsafe_allow_html=True)
