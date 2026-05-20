"""
dashboard/components/upload_widget.py
───────────────────────────────────────
Streamlit component: premium styled file upload area with image preview.
"""
import streamlit as st
from PIL import Image


def render_upload_section():
    """
    Renders the upload widget and returns (image, uploaded_file) or (None, None).

    Returns
    -------
    image : PIL.Image or None
    uploaded_file : UploadedFile or None
    """
    st.markdown(
        '<div class="card-title">📂 Patient Imaging</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop Chest X-Ray here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Accepted formats: JPG, JPEG, PNG. Max size: 200 MB.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded X-Ray", use_container_width=True)

        # Show basic image metadata
        w, h = image.size
        st.markdown(
            f"""
            <div style="display:flex; gap:12px; margin-top:8px;">
                <div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.15);
                            border-radius:8px; padding:6px 14px; font-size:0.78rem; color:#94a3b8;">
                    📐 {w} × {h} px
                </div>
                <div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.15);
                            border-radius:8px; padding:6px 14px; font-size:0.78rem; color:#94a3b8;">
                    🗂 {uploaded_file.type.split("/")[-1].upper()}
                </div>
                <div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.15);
                            border-radius:8px; padding:6px 14px; font-size:0.78rem; color:#94a3b8;">
                    💾 {uploaded_file.size / 1024:.1f} KB
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        return image, uploaded_file

    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:60px 20px; color:#334155;">
            <div style="font-size:3rem;">🫁</div>
            <div style="margin-top:12px; font-size:0.9rem;">Upload a Chest X-Ray to begin</div>
            <div style="font-size:0.75rem; margin-top:6px;">Supports JPG · JPEG · PNG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return None, None
