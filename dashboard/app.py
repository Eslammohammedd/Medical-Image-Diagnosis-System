import streamlit as st
import os
import io
from PIL import Image
from dotenv import load_dotenv

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChestX-Ray AI | Clinical Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Environment ─────────────────────────────────────────────────────────
load_dotenv()

# ─── PREMIUM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #050d1a !important;
    color: #e2e8f0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #060e1c 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.12) !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #38bdf8 !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a2540 50%, #0d1f3c 100%);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60%; left: -5%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
}

/* ── Cards ── */
.card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s ease;
}
.card:hover { border-color: rgba(56,189,248,0.2); }
.card-title {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── Metric Chips ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.metric-chip {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 10px 18px;
    text-align: center;
    flex: 1; min-width: 90px;
}
.metric-chip-value { color: #38bdf8; font-size: 1.3rem; font-weight: 700; }
.metric-chip-label { color: #64748b; font-size: 0.7rem; margin-top: 2px; }

/* ── Disease Bar ── */
.disease-item { margin-bottom: 12px; }
.disease-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
.disease-name { color: #cbd5e1; font-size: 0.85rem; font-weight: 500; }
.disease-score { color: #38bdf8; font-size: 0.85rem; font-weight: 700; }
.disease-bar-bg {
    height: 6px; background: rgba(255,255,255,0.06);
    border-radius: 10px; overflow: hidden;
}
.disease-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
    transition: width 0.8s ease;
}

/* ── Report Box ── */
.report-box {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(56,189,248,0.15);
    border-left: 4px solid #38bdf8;
    border-radius: 12px;
    padding: 24px 28px;
    line-height: 1.8;
    color: #cbd5e1;
    font-size: 0.9rem;
    white-space: pre-wrap;
}

/* ── COVID Result ── */
.covid-card {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 12px;
}
.covid-card.normal {
    background: rgba(34,197,94,0.07);
    border-color: rgba(34,197,94,0.25);
}

/* ── Status Dot ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.status-dot.red { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.25) !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(14,165,233,0.4) !important;
}

/* ── Upload Area ── */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.6) !important;
    border: 2px dashed rgba(56,189,248,0.25) !important;
    border-radius: 14px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.5) !important;
}

/* ── Headers ── */
h1, h2, h3 { color: #f1f5f9 !important; font-weight: 600 !important; }
h2 { font-size: 1.1rem !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Spinner / Info / Alerts ── */
.stSpinner > div { border-top-color: #38bdf8 !important; }
.stAlert { border-radius: 12px !important; border: none !important; }

/* ── Footer Disclaimer ── */
.disclaimer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    padding: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 32px;
}

/* ── Progress Bars (Streamlit native) ── */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #0ea5e9, #6366f1) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helper: render disease bars ─────────────────────────────────────────────
def render_disease_bars(results: dict, top_n: int = 5):
    sorted_r = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_n]
    bars_html = ""
    for name, score in sorted_r:
        pct = score * 100
        color = "#ef4444" if pct > 60 else "#f59e0b" if pct > 30 else "#0ea5e9"
        bars_html += f"""
        <div class="disease-item">
            <div class="disease-header">
                <span class="disease-name">{name}</span>
                <span class="disease-score" style="color:{color}">{pct:.1f}%</span>
            </div>
            <div class="disease-bar-bg">
                <div class="disease-bar-fill" style="width:{pct}%; background:linear-gradient(90deg,{color},{color}aa)"></div>
            </div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🫁 ChestX-Ray AI")
    st.markdown("<hr style='border-color:rgba(56,189,248,0.2)'>", unsafe_allow_html=True)

    analysis_mode = st.radio(
        "**Analysis Mode**",
        ["⚡ Gemini Vision (Fast)", "🔬 Full System (Local + Gemini)"],
        index=0,
        help="Fast mode uses Gemini API only. Full mode adds local DenseNet121 + Grad-CAM."
    )

    st.markdown("<hr style='border-color:rgba(56,189,248,0.1)'>", unsafe_allow_html=True)
    enable_detection = st.checkbox(
        "🎯 Object Detection",
        value=True,
        help="Draw bounding boxes around detected abnormalities (Roboflow)."
    )

    st.markdown("<hr style='border-color:rgba(56,189,248,0.1)'>", unsafe_allow_html=True)

    # API Status
    api_ok = bool(os.getenv("GEMINI_API_KEY"))
    dot_class = "green" if api_ok else "red"
    status_text = "Connected" if api_ok else "Disconnected"
    st.markdown(f"""
    <div style='font-size:0.82rem; color:#94a3b8; margin-bottom:6px;'>API Status</div>
    <div style='font-size:0.9rem;'>
        <span class='status-dot {dot_class}'></span>
        <span style='color:{"#22c55e" if api_ok else "#ef4444"};font-weight:600;'>Gemini {status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem; color:#334155; text-align:center;'>
    v1.0 Gold Release · PyTorch · Streamlit
    </div>""", unsafe_allow_html=True)


# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🏥 CLINICAL AI DASHBOARD</div>
    <h1 class="hero-title">ChestX-Ray Diagnostic Intelligence</h1>
    <p class="hero-subtitle">Triple-AI Pipeline · DenseNet121 · Roboflow OD · Gemini 1.5 Flash</p>
</div>
""", unsafe_allow_html=True)


# ─── Main Layout ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.3], gap="large")

# ── Left: Upload ──────────────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="card-title">📂 Patient Imaging</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop Chest X-Ray here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Run Full Analysis")
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#334155;'>
            <div style='font-size:3rem;'>🫁</div>
            <div style='margin-top:12px; font-size:0.9rem;'>Upload a Chest X-Ray to begin</div>
            <div style='font-size:0.75rem; margin-top:6px;'>Supports JPG, JPEG, PNG</div>
        </div>
        """, unsafe_allow_html=True)


# ── Right: Results ────────────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="card-title">📋 Diagnostic Findings</div>', unsafe_allow_html=True)

    if uploaded_file and 'analyze_btn' in locals() and analyze_btn:

        results = {}

        # ── 1. Local Model ────────────────────────────────────────────────────
        if "Full System" in analysis_mode:
            with st.spinner("Running local DenseNet121 inference..."):
                try:
                    from inference.predictor import ChestXrayPredictor
                    predictor = ChestXrayPredictor()
                    results = predictor.predict(uploaded_file)
                    top_disease = predictor.get_top_class(results)
                    confidence = results[top_disease]

                    st.markdown(f"""
                    <div class="metric-row">
                        <div class="metric-chip">
                            <div class="metric-chip-value">{confidence:.0%}</div>
                            <div class="metric-chip-label">Confidence</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{len(results)}</div>
                            <div class="metric-chip-label">Pathologies Checked</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value" style="font-size:0.9rem">{top_disease[:10]}</div>
                            <div class="metric-chip-label">Top Finding</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Top Disease Probabilities**")
                    render_disease_bars(results, top_n=5)

                except Exception as e:
                    st.warning(f"Local model unavailable (weights not trained yet). Using API mode. `{e}`")

        # ── 2. Gemini Report ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📝 AI Radiology Report**")
        with st.spinner("Gemini 1.5 Flash generating clinical report..."):
            try:
                from inference.gemini_integration import GeminiRadiologyReporter
                reporter = GeminiRadiologyReporter()
                report_text = reporter.generate_report(
                    uploaded_file,
                    results if results else {}
                )
                st.markdown(f'<div class="report-box">{report_text}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="report-box" style="border-left-color:#f59e0b; color:#94a3b8;">
                ⚠️ Gemini API unavailable. Check your GEMINI_API_KEY in .env
                <code style="font-size:0.78rem;">{e}</code>
                </div>""", unsafe_allow_html=True)

        # ── 3. Roboflow Detection ─────────────────────────────────────────────
        # ── 4. Roboflow Detection & Boxes ─────────────────────────────────────
        if enable_detection:
            st.write("")
            st.markdown("**🎯 Abnormality Localization (Red Boxes)**")
            with st.spinner("Locating abnormalities..."):
                try:
                    from inference.roboflow_integration import RoboflowDetector
                    detector = RoboflowDetector()
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='JPEG')
                    img_bytes = img_bytes.getvalue()

                    # 3a. Detection (bounding boxes)
                    detection_results = detector.detect(img_bytes)
                    if detection_results.get('predictions'):
                        detected_img = detector.draw_boxes(image.copy(), detection_results)
                        st.image(detected_img, caption="Abnormality Localization (Red Bounding Boxes)", use_container_width=True)
                        st.success(f"📍 Found {len(detection_results['predictions'])} suspected infection site(s).")
                    else:
                        st.info("✅ No structural abnormalities localized with red boxes.")

                    # 3b. COVID Specialized Classification
                    try:
                        covid_results = detector.classify_covid(img_bytes)
                        preds = covid_results.get('predictions')
                        
                        if preds:
                            # Handle both list and dict formats from Roboflow
                            if isinstance(preds, list) and len(preds) > 0:
                                pred = preds[0]
                                label = pred.get('class', 'Unknown')
                                conf = pred.get('confidence', 0.0)
                            elif isinstance(preds, dict):
                                # Dynamic extraction for dict format
                                label = covid_results.get('top')
                                if not label:
                                    # Find key with highest confidence
                                    label = max(preds.keys(), key=lambda k: preds[k].get('confidence', 0))
                                
                                conf = preds[label].get('confidence', 0.0) if label in preds else 0.0
                            else:
                                label, conf = "Unknown", 0.0

                            card_class = "normal" if "normal" in label.lower() else ""
                            icon = "✅" if "normal" in label.lower() else "🦠"
                            st.markdown(f"""
                            <div class="covid-card {card_class}">
                                <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:4px;">SPECIALIZED ROBOLFLOW CLASSIFICATION</div>
                                <div style="font-size:1.1rem;font-weight:700;">{icon} {label}</div>
                                <div style="font-size:0.85rem;color:#64748b;">Confidence: {conf:.1%}</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ COVID Model: Response received but no predictions found.")
                    except Exception as covid_e:
                        import traceback
                        full_err = traceback.format_exc()
                        st.warning(f"⚠️ COVID Model Error: {str(covid_e) if str(covid_e) else 'Unknown error'}")
                        with st.expander("Show Technical Error"):
                            st.code(full_err)

                except Exception as e:
                    st.error(f"🔴 Detection error: {e}")

        # ── 5. Diagnostic Flowchart (Moved to end for full context) ───────────
        st.write("") # Clean spacing
        st.markdown("**🛤️ Diagnostic Decision Flow**")
        
        # Calculate Infection Status
        current_top = top_disease if 'top_disease' in locals() else "Normal"
        is_normal = "Normal" in current_top
        status_color = "#22c55e" if is_normal else "#ef4444"
        status_text = "NORMAL" if is_normal else "INFECTION DETECTED"
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05);">
            <div style="text-align: center; flex: 1;">
                <div style="color: #64748b; font-size: 0.7rem; margin-bottom: 5px;">INPUT</div>
                <div style="color: #38bdf8; font-weight: 700;">X-Ray Image</div>
            </div>
            <div style="color: #334155;">➜</div>
            <div style="text-align: center; flex: 1;">
                <div style="color: #64748b; font-size: 0.7rem; margin-bottom: 5px;">LOCAL AI</div>
                <div style="color: {status_color}; font-weight: 700;">{status_text}</div>
            </div>
            <div style="color: #334155;">➜</div>
            <div style="text-align: center; flex: 1;">
                <div style="color: #64748b; font-size: 0.7rem; margin-bottom: 5px;">CLASSIFICATION</div>
                <div style="color: #818cf8; font-weight: 700;">{current_top}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


        # ── 6. Grad-CAM ───────────────────────────────────────────────────────
        if 'top_disease' in locals() and top_disease:
            st.write("") # Clean spacing
            st.markdown("**🗺️ Explainability Map (Grad-CAM)**")
            st.caption(f"AI attention area for: **{top_disease}**")
            st.image(image, caption="Grad-CAM Overlay (Training required for real heatmap)", use_container_width=True)

    elif not uploaded_file:
        st.markdown("""
        <div style='padding:40px 20px; text-align:center; color:#334155;'>
            <div style='font-size:2rem;'>📋</div>
            <div style='margin-top:10px; font-size:0.9rem;'>Results will appear here after analysis</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer Disclaimer ────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This system is AI-assisted and intended for research purposes only.
    All outputs must be reviewed by a licensed radiologist before any clinical use.
    · ChestX-Ray AI v1.0 · Eslam Mohammed
</div>
""", unsafe_allow_html=True)
