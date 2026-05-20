import os
import time
import requests
import base64
from PIL import Image
import io
from dotenv import load_dotenv

class GeminiRadiologyReporter:
    """
    Integrates with Google Gemini API using direct REST calls for maximum reliability.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file.")
        
        # Test models in order
        self.model_name = "gemini-2.5-flash" # The available model for this API key

    def generate_report(self, image_file, model_predictions: dict) -> str:
        """
        Sends the X-Ray image + predictions to Gemini via direct REST API.
        """
        # 1. Prepare Image
        img = Image.open(image_file).convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 2. Build Prompt
        pred_context = "\n".join([f"- {k}: {v:.1%}" for k, v in model_predictions.items() if v > 0.1])
        prompt = f"Analyze this Chest X-Ray. Context: {pred_context}. Write a professional Radiology Report with 'CLINICAL OBSERVATIONS' and 'IMPRESSION' sections. End with 'AI-Assisted Analysis'."

        # 3. Direct REST Call
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }]
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                err_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Gemini API Error {response.status_code}: {err_msg}")
        except Exception as e:
            raise Exception(f"Connection failed: {str(e)}")
