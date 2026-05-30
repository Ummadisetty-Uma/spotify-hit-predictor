"""
================================================
  Spotify Hit Predictor — Streamlit App (Task 7)
================================================
  Run with:  streamlit run app.py
================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import json 
import os

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%); }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: -1px;
    }
    .sub-header {
        color: #b3b3b3;
        text-align: center;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(29,185,84,0.3);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #1DB954; }
    .metric-label { font-size: 0.78rem; color: #b3b3b3; text-transform: uppercase; letter-spacing: 1px; }

    /* Result banners */
    .hit-banner {
        background: linear-gradient(135deg, #1DB954, #17a348);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(29,185,84,0.4);
    }
    .nohit-banner {
        background: linear-gradient(135deg, #e05c5c, #c0392b);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(224,92,92,0.4);
    }
    .banner-emoji  { font-size: 3.5rem; }
    .banner-title  { font-size: 2rem; font-weight: 900; color: white; margin: 8px 0; }
    .banner-sub    { font-size: 1rem; color: rgba(255,255,255,0.85); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(13,13,13,0.95);
        border-right: 1px solid rgba(29,185,84,0.2);
    }
    .sidebar-title { color: #1DB954; font-size: 1.1rem; font-weight: 700; margin-bottom: 5px; }

    /* Sliders — label color */
    .stSlider label { color: #e0e0e0 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #1DB954, #17a348);
        color: white !important;
        border: none;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 14px 40px;
        width: 100%;
        cursor: pointer;
        letter-spacing: 0.5px;
        transition: transform 0.15s;
    }
    .stButton > button:hover { transform: scale(1.02); }

    /* Progress bar override */
    .stProgress > div > div { background-color: #1DB954 !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD ARTEFACTS ──────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_artifacts():
    scaler        = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
    model         = joblib.load(os.path.join(BASE, "models/best_model.pkl"))
    feature_names = joblib.load(os.path.join(BASE, "models/feature_names.pkl"))
    with open(os.path.join(BASE, "models/results.json")) as f:
        results = json.load(f)
    return scaler, model, feature_names, results

scaler, model, feature_names, results = load_artifacts()

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🎵 Spotify Hit Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter a song\'s audio features to predict whether it will be a HIT</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ─── SIDEBAR — MODEL PERFORMANCE ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Performance")
    st.markdown("---")

    best = results["best_model"]
    st.markdown(f"**Best Model:** `{best}`")
    st.markdown("")

    model_metrics = results[best]
    for metric, value in model_metrics.items():
        st.markdown(f"**{metric}**: `{value:.4f}`")
        st.progress(value)

    st.markdown("---")
    st.markdown("### 📊 All Models")
    model_names = ["Logistic Regression", "SVM (RBF Kernel)", "Random Forest"]
    for m in model_names:
        if m in results and isinstance(results[m], dict):
            f1 = results[m].get("F1-Score", 0)
            icon = "🏆" if m == best else "📈"
            st.markdown(f"{icon} **{m}** — F1: `{f1:.4f}`")

    st.markdown("---")
    st.caption("🎵 Spotify Hit Predictor | End-to-End ML Project")

# ─── MAIN PANEL — FEATURE INPUTS ─────────────────────────────────────────────
st.markdown("### 🎚️  Song Audio Features")
st.markdown("Adjust the sliders below to match the song's audio characteristics.")
st.markdown("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🕺 Rhythm & Mood**")
    danceability = st.slider("Danceability",  0.0, 1.0, 0.55, 0.01,
                             help="How suitable the track is for dancing (0=low, 1=high)")
    valence      = st.slider("Valence",        0.0, 1.0, 0.48, 0.01,
                             help="Musical positivity (0=sad/angry, 1=happy/euphoric)")
    energy       = st.slider("Energy",         0.0, 1.0, 0.70, 0.01,
                             help="Intensity and activity level (0=calm, 1=very energetic)")
    liveness     = st.slider("Liveness",       0.0, 1.0, 0.20, 0.01,
                             help="Probability of a live audience in recording")

with col2:
    st.markdown("**🎸 Acoustics & Voice**")
    acousticness     = st.slider("Acousticness",     0.0, 1.0, 0.21, 0.01,
                                 help="Confidence measure for whether the track is acoustic")
    speechiness      = st.slider("Speechiness",      0.0, 1.0, 0.09, 0.01,
                                 help="Proportion of spoken words (>0.66 = pure speech)")
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.15, 0.01,
                                 help="Likelihood of no vocals; >0.5 = instrumental")
    loudness         = st.slider("Loudness (dB)",  -60.0, 5.0, -7.0, 0.1,
                                 help="Overall loudness of the track in decibels")

with col3:
    st.markdown("**🎵 Timing & Structure**")
    tempo          = st.slider("Tempo (BPM)",  46.0, 215.0, 122.0, 0.5,
                               help="Estimated beats per minute")
    duration_ms    = st.slider("Duration (ms)", 15920, 600000, 230000, 1000,
                               help="Song duration in milliseconds (230000 ≈ 3m50s)")
    chorus_hit     = st.slider("Chorus Hit (s)", 0.0, 263.0, 37.0, 0.5,
                               help="When the first chorus drops (seconds into the song)")
    sections       = st.slider("Sections",      1, 30, 10, 1,
                               help="Number of musical sections (intro/verse/chorus/bridge etc.)")

st.markdown("")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**🎹 Musical Key**")
    key            = st.selectbox("Key (0=C … 11=B)", list(range(12)), index=5,
                                  help="Musical key the track is in (Pitch Class notation)")
    mode           = st.radio("Mode", ["Minor (0)", "Major (1)"], index=1,
                              help="Modality of the track")
with col_b:
    st.markdown("**⏱️ Time Signature**")
    time_signature = st.selectbox("Time Signature", [0, 1, 2, 3, 4, 5], index=4,
                                  help="Beats per bar (3 = 3/4, 4 = 4/4)")

mode_val = 1 if "Major" in mode else 0

# ─── PREDICT ─────────────────────────────────────────────────────────────────
st.markdown("---")
predict_col, _ = st.columns([1, 2])

with predict_col:
    predict_btn = st.button("🎯  Predict Hit or Not Hit")

if predict_btn:
    # Assemble input in exact feature order used during training
    input_dict = {
        "danceability":     danceability,
        "energy":           energy,
        "key":              key,
        "loudness":         loudness,
        "mode":             mode_val,
        "speechiness":      speechiness,
        "acousticness":     acousticness,
        "instrumentalness": instrumentalness,
        "liveness":         liveness,
        "valence":          valence,
        "tempo":            tempo,
        "duration_ms":      duration_ms,
        "time_signature":   time_signature,
        "chorus_hit":       chorus_hit,
        "sections":         sections,
    }

    input_df    = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]
    hit_prob     = probability[1]
    nohit_prob   = probability[0]

    st.markdown("---")
    st.markdown("### 🔮 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div class="hit-banner">
            <div class="banner-emoji">🏆</div>
            <div class="banner-title">THIS SONG IS A HIT!</div>
            <div class="banner-sub">Confidence: {hit_prob*100:.1f}% probability of becoming a hit</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="nohit-banner">
            <div class="banner-emoji">📉</div>
            <div class="banner-title">NOT A HIT</div>
            <div class="banner-sub">Confidence: {nohit_prob*100:.1f}% probability of not charting</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("#### 📊 Probability Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🏆 Hit Probability",     f"{hit_prob*100:.1f}%")
        st.progress(float(hit_prob))
    with c2:
        st.metric("📉 Not-Hit Probability", f"{nohit_prob*100:.1f}%")
        st.progress(float(nohit_prob))

    # Feature summary
    st.markdown("#### 🎚️ Your Song's Audio Profile")
    profile_df = pd.DataFrame({
        "Feature": list(input_dict.keys()),
        "Value":   [round(v, 4) for v in input_dict.values()]
    })
    st.dataframe(profile_df.set_index("Feature").T, use_container_width=True)

    # Advice
    st.markdown("#### 💡 Insight")
    insights = []
    if danceability > 0.7:
        insights.append("✅ High danceability — club-friendly tracks tend to chart well.")
    if instrumentalness > 0.5:
        insights.append("⚠️ High instrumentalness — most chart hits feature vocals.")
    if loudness < -15:
        insights.append("⚠️ Low loudness — modern hits are typically mastered louder.")
    if energy > 0.8:
        insights.append("✅ High energy — energetic songs perform well on streaming.")
    if not insights:
        insights.append("🎵 The song has a balanced audio profile.")
    for ins in insights:
        st.info(ins)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><sub>🎵 Spotify Hit Predictor · End-to-End ML Project · "
    "Model: SVM (RBF Kernel) · Dataset: Spotify Hit Predictor (5,872 songs)</sub></center>",
    unsafe_allow_html=True,
)
