"""
dashboard/components/heatmap_viewer.py
────────────────────────────────────────
Streamlit component: renders a Grad-CAM heatmap overlay on top of the
original X-Ray image.  Falls back gracefully when no trained weights exist.
"""
import streamlit as st
import numpy as np
from PIL import Image
import io


def show_gradcam_overlay(original_image: Image.Image, heatmap: np.ndarray | None = None):
    """
    Displays the original image with an optional Grad-CAM heatmap overlay.

    Parameters
    ----------
    original_image : PIL.Image
        The uploaded chest X-Ray.
    heatmap : np.ndarray or None
        A (H, W) float array in [0, 1] produced by GradCAM.generate_heatmap().
        If None a placeholder is shown instead.
    """
    if heatmap is not None:
        try:
            import cv2
            from inference.gradcam import overlay_heatmap

            # Convert PIL → numpy BGR for cv2
            img_np = np.array(original_image.convert('RGB'))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            heatmap_resized = cv2.resize(heatmap.astype(np.float32),
                                         (img_bgr.shape[1], img_bgr.shape[0]))
            overlayed = overlay_heatmap(img_bgr, heatmap_resized)
            overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)

            st.image(
                overlayed_rgb,
                caption="🗺️ Grad-CAM Heatmap – AI attention areas highlighted in warm colours",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Grad-CAM rendering failed: {e}")
            st.image(original_image, caption="Original X-Ray (Grad-CAM unavailable)",
                     use_container_width=True)
    else:
        st.image(
            original_image,
            caption="📷 Grad-CAM Overlay (train the model to enable real heatmaps)",
            use_container_width=True,
        )
        st.caption(
            "ℹ️ Once model weights are trained, the heatmap will highlight "
            "the exact regions the AI used to make its decision."
        )
