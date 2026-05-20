"""
dashboard/components/report_generator.py
─────────────────────────────────────────
Streamlit component: wraps the Gemini radiology report generation and
formats the output inside a premium styled box.
"""
import streamlit as st
from PIL import Image


def render_report_section(uploaded_file, model_predictions: dict = None):
    """
    Calls GeminiRadiologyReporter and renders the resulting report.

    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile
        The raw uploaded file object.
    model_predictions : dict, optional
        {class_name: probability} from the local DenseNet model (can be empty).
    """
    if model_predictions is None:
        model_predictions = {}

    st.markdown("**📝 AI Radiology Report**")
    with st.spinner("Gemini 1.5 Flash generating clinical report…"):
        try:
            from inference.gemini_integration import GeminiRadiologyReporter
            reporter = GeminiRadiologyReporter()
            report_text = reporter.generate_report(uploaded_file, model_predictions)

            # Render inside styled box
            st.markdown(
                f'<div class="report-box">{report_text}</div>',
                unsafe_allow_html=True,
            )

            # Download button
            st.download_button(
                label="⬇️ Download Report (.txt)",
                data=report_text,
                file_name="radiology_report.txt",
                mime="text/plain",
            )

        except ValueError as e:
            # Missing API key
            st.markdown(
                f"""<div class="report-box" style="border-left-color:#ef4444; color:#94a3b8;">
                🔴 <strong>API Key Missing</strong><br>{e}
                </div>""",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(
                f"""<div class="report-box" style="border-left-color:#f59e0b; color:#94a3b8;">
                ⚠️ Gemini API unavailable. Check your <code>GEMINI_API_KEY</code> in <code>.env</code><br>
                <code style="font-size:0.78rem;">{e}</code>
                </div>""",
                unsafe_allow_html=True,
            )
