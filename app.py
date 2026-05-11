"""
Sonic Vibe — Music Sentiment Analysis Dashboard
================================================
A premium Streamlit interface for the Music Emotional Vibe pipeline.
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import os
import tempfile
import plotly.graph_objects as go
import librosa

from extract import extract_features, get_feature_names, SAMPLE_RATE, DURATION
from train import (
    load_model, train_model, VIBE_LABELS, VIBE_DISPLAY,
    VIBE_COLORS, VIBE_EMOJIS, MODEL_PATH,
)
from agent import explain_prediction, get_shap_details, contrast_analysis, generate_song_names

# ----------------------------------------------
# Page Configuration
# ----------------------------------------------
st.set_page_config(
    page_title="Sonic Vibe — Music Sentiment Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------
# Custom CSS
# ----------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* -- Global -- */
.stApp {
    background: linear-gradient(170deg, #0a0a0f 0%, #12121f 40%, #0d1117 100%);
    font-family: 'Outfit', sans-serif;
}
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* -- Sidebar (Hidden) -- */
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* -- Hero -- */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #a78bfa 0%, #818cf8 30%, #f472b6 70%, #fb923c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.45);
    font-weight: 300;
    text-align: center;
    margin-top: 4px;
}
.hero-architect {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: rgba(255,255,255,0.25);
    text-align: center;
    margin-top: 14px;
}

/* -- Section Headers -- */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: rgba(255,255,255,0.3);
    margin-bottom: 12px;
    padding-left: 2px;
}

/* -- Glass Card -- */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 16px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.1);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* -- Vibe Result Card -- */
.vibe-result {
    text-align: center;
    padding: 40px 28px;
    border-radius: 24px;
    position: relative;
    overflow: hidden;
}
.vibe-result::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 24px;
    padding: 2px;
    background: var(--vibe-gradient);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.6;
}
.vibe-emoji { font-size: 3.5rem; margin-bottom: 8px; }
.vibe-name {
    font-size: 2rem;
    font-weight: 700;
    margin: 8px 0 4px;
}
.vibe-confidence {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.5);
    font-weight: 400;
}

/* -- Equalizer Animation -- */
.eq-container {
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 4px;
    height: 100px;
    margin-bottom: 20px;
}
.eq-bar {
    width: 6px;
    border-radius: 3px;
    animation: eq-bounce var(--dur) ease-in-out infinite alternate;
    opacity: 0.8;
}
@keyframes eq-bounce {
    0% { height: 10%; }
    100% { height: 100%; }
}

/* -- Landing Vibe Cards -- */
.vibe-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin-top: 20px;
}
.vibe-mini {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.3s ease;
}
.vibe-mini:hover {
    transform: translateY(-3px);
    border-color: rgba(255,255,255,0.12);
}
.vibe-mini-emoji { font-size: 1.8rem; }
.vibe-mini-label {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 8px;
    color: rgba(255,255,255,0.8);
}
.vibe-mini-desc {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.35);
    margin-top: 4px;
}

/* -- SHAP Detail Row -- */
.shap-row {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    border-radius: 12px;
    background: rgba(255,255,255,0.02);
    margin-bottom: 8px;
    gap: 16px;
}
.shap-fname {
    font-size: 0.8rem;
    font-weight: 600;
    color: rgba(255,255,255,0.6);
    min-width: 130px;
    font-family: 'Courier New', monospace;
}
.shap-desc {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.85);
    flex: 1;
}
.shap-val {
    font-size: 0.8rem;
    font-weight: 600;
    min-width: 65px;
    text-align: right;
    font-family: 'Courier New', monospace;
}

/* -- Content Card -- */
.content-block {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 12px;
    white-space: pre-wrap;
    font-size: 0.92rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.8);
}
.content-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

/* -- Misc -- */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 32px 0;
}
.stAudio > div { border-radius: 12px; overflow: hidden; }

/* -- Expander Visibility Fix -- */
[data-testid="stExpanderHeader"] p, 
[data-testid="stExpanderHeader"] span, 
[data-testid="stExpanderHeader"] summary {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

[data-testid="stExpanderDetails"] p, 
[data-testid="stExpanderDetails"] span, 
[data-testid="stExpanderDetails"] div {
    color: rgba(255, 255, 255, 0.9) !important;
}

/* -- Song Namer Cards -- */
.song-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}
.song-card:hover {
    background: rgba(255,255,255,0.04);
    border-color: rgba(255,255,255,0.1);
}
.song-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #fff, rgba(255,255,255,0.5));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.song-why {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    line-height: 1.6;
    font-family: 'Outfit', sans-serif;
}
.tech-tag {
    font-family: 'Courier New', monospace;
    color: #a78bfa;
    font-weight: 600;
}

/* -- Custom Button -- */
div.stButton > button {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    filter: brightness(1.1) !important;
}

/* -- Streamlit File Uploader Hack -- */
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.02) !important;
    padding: 10px !important;
    border-radius: 20px !important;
}
[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: 1px dashed rgba(167, 139, 250, 0.3) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(255, 255, 255, 0.02) !important;
    border-color: rgba(167, 139, 250, 0.6) !important;
}

/* FIX: Absolute visibility for uploader instructions */
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: #ffffff !important;
}

/* Target any remaining text inside the dropzone */
[data-testid="stFileUploadDropzone"] div {
    color: #ffffff !important;
}

[data-testid="stFileUploadDropzone"] button {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploadDropzone"] button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* -- General Contrast Boost -- */
.vibe-mini-desc {
    color: rgba(255,255,255,0.6) !important;
}
.vibe-confidence {
    color: rgba(255,255,255,0.7) !important;
}

/* -- Footer -- */
.footer {
    text-align: center;
    color: rgba(255,255,255,1.0); /* High visibility white */
    font-size: 0.7rem;
    margin-top: 80px;
    padding-bottom: 40px;
    letter-spacing: 2px;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------
# Plotting helpers
# ----------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif"),
    showlegend=False,
)


def plot_waveform(y, sr, color="#a78bfa"):
    hop = max(1, len(y) // 3000)
    y_ds, t_ds = y[::hop], np.arange(len(y[::hop])) * hop / sr
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig = go.Figure(go.Scatter(
        x=t_ds, y=y_ds, mode="lines",
        line=dict(color=color, width=0.7),
        fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.1)", hoverinfo="skip",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=180, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.03)", zeroline=False),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.03)", zeroline=False, showticklabels=False),
    )
    return fig


def plot_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = go.Figure(go.Heatmap(z=S_dB, colorscale="Magma", showscale=False))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=180, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="", showticklabels=False),
        yaxis=dict(title="", showticklabels=False),
    )
    return fig


def plot_probabilities(labels, probs):
    pairs = sorted(zip(labels, probs), key=lambda x: x[1])
    names = [VIBE_DISPLAY[l] for l, _ in pairs]
    vals = [p for _, p in pairs]
    colors = [VIBE_COLORS[l] for l, _ in pairs]
    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation="h",
        marker=dict(color=colors, cornerradius=5),
        text=[f"{v:.0%}" for v in vals], textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=12),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=280, margin=dict(l=10, r=50, t=5, b=5),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, max(vals) * 1.35]),
        yaxis=dict(gridcolor="rgba(255,255,255,0.02)"),
    )
    return fig


# ----------------------------------------------
# Landing Page (no file uploaded)
# ----------------------------------------------
if "uploaded_file_state" not in st.session_state:
    st.session_state.uploaded_file_state = None

# Centered layout container
_, main_col, _ = st.columns([1, 3, 1])

with main_col:
    # If we are on the results page but want to go back
    if st.button("⬅️ Back to Landing", key="back_btn") if "uploaded_file_state" in st.session_state and st.session_state.uploaded_file_state else False:
        st.session_state.uploaded_file_state = None
        st.rerun()

    if st.session_state.uploaded_file_state is None:
        st.markdown('<div class="hero-title">Sonic Vibe</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Decode the emotional DNA of any track — powered by AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-architect">ARCHITECTED BY SHAYAN ANSARI</div>', unsafe_allow_html=True)

        # ── Hero Animation ──
        bars_html = ""
        colors_cycle = ["#a78bfa", "#818cf8", "#f472b6", "#fb923c", "#34d399", "#fbbf24"]
        for i in range(28):
            c = colors_cycle[i % len(colors_cycle)]
            dur = round(0.6 + (i % 7) * 0.12, 2)
            bars_html += f'<div class="eq-bar" style="background:{c};--dur:{dur}s;animation-delay:{i*0.04}s;"></div>'
        st.markdown(f'<div class="eq-container" style="margin-top:40px; margin-bottom:40px;">{bars_html}</div>', unsafe_allow_html=True)

        # ── Action Area (Uploader + Retrain) ──
        st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
        
        file = st.file_uploader("Upload a track", type=["wav", "mp3", "ogg", "flac", "m4a", "aac", "wma", "mpeg"], label_visibility="collapsed", key="main_uploader")
        if file:
            st.session_state.uploaded_file_state = file
            st.rerun()
            
        st.markdown('<p style="text-align:center; color:white; font-size:0.75rem; margin-top:-10px; margin-bottom:25px;">WAV · MP3 · OGG · FLAC · M4A · AAC · WMA · MPEG</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            if st.button("🔄  Retrain Model", use_container_width=True, key="main_retrain"):
                with st.spinner("Training…"):
                    train_model()
                st.success("Ready!")
        with c2:
            if os.path.exists(MODEL_PATH):
                st.markdown('<div style="margin-top:10px; color:#34d399; font-size:0.85rem; font-weight:600; text-align:center;">✅ Model Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="margin-top:10px; color:#fbbf24; font-size:0.85rem; font-weight:600; text-align:center;">⚠️ No Model</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label" style="text-align:center; width:100%;">Five Emotional Vibes</div>', unsafe_allow_html=True)

        vibe_descs = {
            "melancholic": "Low energy, dark tones",
            "high_energy": "Fast tempo, bright sound",
            "ambient": "Calm textures, spacious",
            "aggressive": "Loud, harsh, intense",
            "nostalgic": "Warm, mid-tempo glow",
        }
        grid_html = '<div class="vibe-grid">'
        for label in VIBE_LABELS:
            emoji = VIBE_EMOJIS[label]
            color = VIBE_COLORS[label]
            display = VIBE_DISPLAY[label].split(" / ")[0]
            mood = VIBE_DISPLAY[label].split(" / ")[1]
            desc = vibe_descs[label]
            grid_html += f'''<div class="vibe-mini" style="border-bottom:2px solid {color}33;">
                <div class="vibe-mini-emoji">{emoji}</div>
                <div class="vibe-mini-label" style="color:{color};">{display}</div>
                <div style="font-size:0.78rem;color:rgba(255,255,255,0.55);margin-top:2px;font-weight:500;">{mood}</div>
                <div class="vibe-mini-desc">{desc}</div>
            </div>'''
        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="glass-card" style="text-align:center;">
                <div style="font-size:1.5rem;">📤</div>
                <div style="font-size:0.85rem;font-weight:600;color:rgba(255,255,255,0.8);margin-top:8px;">Upload</div>
                <div style="font-size:0.75rem;color:rgba(255,255,255,0.35);margin-top:4px;">Drop any audio file</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="glass-card" style="text-align:center;">
                <div style="font-size:1.5rem;">🧠</div>
                <div style="font-size:0.85rem;font-weight:600;color:rgba(255,255,255,0.8);margin-top:8px;">Analyze</div>
                <div style="font-size:0.75rem;color:rgba(255,255,255,0.35);margin-top:4px;">28 audio features extracted</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="glass-card" style="text-align:center;">
                <div style="font-size:1.5rem;">✍️</div>
                <div style="font-size:0.85rem;font-weight:600;color:rgba(255,255,255,0.8);margin-top:8px;">Create</div>
                <div style="font-size:0.75rem;color:rgba(255,255,255,0.35);margin-top:4px;">AI-generated song titles</div>
            </div>""", unsafe_allow_html=True)

        st.stop()

    else:
        uploaded_file = st.session_state.uploaded_file_state
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<div class="hero-title" style="font-size:2rem;">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hero-sub">🎧 {uploaded_file.name}</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-architect">ARCHITECTED BY SHAYAN ANSARI</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Save uploaded file to temp
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            # -- Load audio & extract features --
            with st.spinner("Extracting audio features…"):
                y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
                feature_vector = extract_features(tmp_path)

            # -- Ensure model exists --
            if not os.path.exists(MODEL_PATH):
                with st.spinner("Training model (first run)…"):
                    model = train_model()
            else:
                model = load_model()

            # -- Predict --
            predicted_vibe = model.predict(feature_vector.reshape(1, -1))[0]
            proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
            class_labels = list(model.classes_)
            
            win_idx = class_labels.index(predicted_vibe)
            confidence = float(proba[win_idx])
            
            # Runner-up logic
            prob_pairs = sorted(zip(class_labels, proba), key=lambda x: x[1], reverse=True)
            winner_label, winner_prob = prob_pairs[0]
            runner_up_label, runner_up_prob = prob_pairs[1]
            
            vibe_display = VIBE_DISPLAY[predicted_vibe]
            runner_up_display = VIBE_DISPLAY[runner_up_label]
            vibe_color = VIBE_COLORS[predicted_vibe]
            vibe_emoji = VIBE_EMOJIS[predicted_vibe]

            # -- SHAP --
            with st.spinner("Computing SHAP explanations…"):
                shap_summary = explain_prediction(model, feature_vector, predicted_vibe, top_k=5)
                shap_details = get_shap_details(model, feature_vector, predicted_vibe, top_k=5)
                contrast_details = contrast_analysis(model, feature_vector, winner_label, runner_up_label, top_k=3)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # -- Audio Player + Waveform --
            st.markdown('<div class="section-label">Audio Preview</div>', unsafe_allow_html=True)
            st.audio(tmp_path)

            wave_tab, spec_tab = st.tabs(["🌊 Waveform", "🔥 Spectrogram"])
            with wave_tab:
                st.plotly_chart(plot_waveform(y, sr, vibe_color), use_container_width=True, config={"displayModeBar": False})
            with spec_tab:
                st.plotly_chart(plot_spectrogram(y, sr), use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # -- Vibe Result --
            r, g, b = int(vibe_color[1:3], 16), int(vibe_color[3:5], 16), int(vibe_color[5:7], 16)
            gradient = f"linear-gradient(135deg, rgba({r},{g},{b},0.15), rgba({r},{g},{b},0.03))"
            border_grad = f"linear-gradient(135deg, {vibe_color}, transparent, {vibe_color})"

            col_main, col_prob = st.columns([1, 1])

            with col_main:
                st.markdown('<div class="section-label">Detected Vibe</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="vibe-result" style="background:{gradient};--vibe-gradient:{border_grad};">
                    <div class="vibe-emoji">{vibe_emoji}</div>
                    <div class="vibe-name" style="color:{vibe_color};">{vibe_display}</div>
                    <div class="vibe-confidence">{confidence:.0%} confidence</div>
                </div>
                """, unsafe_allow_html=True)

            with col_prob:
                st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_probabilities(class_labels, proba), use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # -- Contrast Analysis --
            st.markdown(f'<div class="section-label">Contrast Analysis — Winner vs Runner-up</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:rgba(255,255,255,0.4);font-size:0.85rem;margin-bottom:16px;">Why <b>{vibe_display.split(" / ")[0]}</b> beat <b>{runner_up_display.split(" / ")[0]}</b> ({runner_up_prob:.0%} match)</p>', unsafe_allow_html=True)
            
            col_c1, col_c2, col_c3 = st.columns(3)
            cols = [col_c1, col_c2, col_c3]
            for i, d in enumerate(contrast_details):
                with cols[i]:
                    st.markdown(f"""
                    <div class="glass-card" style="padding:20px; border-top: 3px solid {vibe_color};">
                        <div style="font-size:0.7rem; text-transform:uppercase; color:rgba(255,255,255,0.3); letter-spacing:1px;">{d["feature"]}</div>
                        <div style="font-size:0.9rem; font-weight:600; color:{vibe_color}; margin:4px 0;">{d["descriptor"]}</div>
                        <div style="font-size:0.75rem; color:rgba(255,255,255,0.5); line-height:1.4;">
                            This feature provided a <b>+{d["diff"]:.3f}</b> net advantage over the runner-up vibe.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # -- LLM Song Namer --
            st.markdown('<div class="section-label">Agentic Song Namer — Ollama / Llama 3.2</div>', unsafe_allow_html=True)

            if st.button("✨  Generate Song Titles", use_container_width=True):
                with st.spinner("Curating titles based on sonic profile…"):
                    response = generate_song_names(vibe_display, runner_up_display, shap_summary)
                
                # Simple parsing for better display
                import re
                # Look for patterns like "1. Title" and "Why: Reason"
                titles = re.findall(r'### \d+\. (.*?)\n\*\*Why:\*\* (.*?)(?=\n###|\Z)', response, re.DOTALL)
                
                if titles:
                    for title, why in titles:
                        st.markdown(f"""
                        <div class="song-card">
                            <div class="song-title">{title.strip('"')}</div>
                            <div class="song-why"><span class="tech-tag">WHY:</span> {why.strip()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Fallback if parsing fails
                    st.markdown(f'<div class="content-block">{response}</div>', unsafe_allow_html=True)

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# -- Footer --
st.markdown("""
<div class="footer">
    © 2026 SONIC VIBE<br>
    ENGINEERED WITH OLLAMA LLM & EXPLAINABLE AI (SHAP)
</div>
""", unsafe_allow_html=True)
