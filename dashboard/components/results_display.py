"""
dashboard/components/results_display.py
──────────────────────────────────────────
Streamlit component: renders the classification results as metric chips
and animated progress bars.
"""
import streamlit as st


def render_metric_chips(top_disease: str, confidence: float, num_classes: int):
    """Renders the three KPI chips at the top of the results panel."""
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-value">{confidence:.0%}</div>
                <div class="metric-chip-label">Confidence</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-value">{num_classes}</div>
                <div class="metric-chip-label">Pathologies Checked</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-value" style="font-size:0.9rem">{top_disease[:10]}</div>
                <div class="metric-chip-label">Top Finding</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disease_bars(results: dict, top_n: int = 5):
    """
    Renders coloured probability bars for the top-N predicted diseases.

    Colour coding:
        red   → > 60 % (high risk)
        amber → 30–60 % (moderate)
        blue  → < 30 % (low / normal)
    """
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
                <div class="disease-bar-fill"
                     style="width:{pct}%; background:linear-gradient(90deg,{color},{color}aa)">
                </div>
            </div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)


def render_diagnostic_flowchart(top_disease: str):
    """Renders the three-step diagnostic decision flow."""
    is_normal    = "Normal" in top_disease
    status_color = "#22c55e" if is_normal else "#ef4444"
    status_text  = "NORMAL" if is_normal else "INFECTION DETECTED"

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    background:rgba(255,255,255,0.03); padding:20px; border-radius:15px;
                    border:1px solid rgba(255,255,255,0.05);">
            <div style="text-align:center; flex:1;">
                <div style="color:#64748b; font-size:0.7rem; margin-bottom:5px;">INPUT</div>
                <div style="color:#38bdf8; font-weight:700;">X-Ray Image</div>
            </div>
            <div style="color:#334155;">➜</div>
            <div style="text-align:center; flex:1;">
                <div style="color:#64748b; font-size:0.7rem; margin-bottom:5px;">LOCAL AI</div>
                <div style="color:{status_color}; font-weight:700;">{status_text}</div>
            </div>
            <div style="color:#334155;">➜</div>
            <div style="text-align:center; flex:1;">
                <div style="color:#64748b; font-size:0.7rem; margin-bottom:5px;">CLASSIFICATION</div>
                <div style="color:#818cf8; font-weight:700;">{top_disease}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
