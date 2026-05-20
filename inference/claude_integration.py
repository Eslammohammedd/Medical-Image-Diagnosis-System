import os
import base64
import anthropic
from dotenv import load_dotenv

class ClaudeRadiologyReporter:
    """
    Integrates with Anthropic Claude Vision API to generate professional radiology reports.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ CLAUDE_API_KEY not found in environment variables.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_report(self, image_path, model_predictions, heatmap_path=None):
        """
        Sends the X-Ray (and optionally heatmap) + local predictions to Claude.
        """
        base64_image = self.encode_image(image_path)
        
        # Prepare context from local model
        pred_context = "\n".join([f"- {d}: {s:.2%}" for d, s in model_predictions.items() if s > 0.1])
        
        prompt = f"""
        You are a senior consultant radiologist. Analyze this Chest X-Ray image.
        
        LOCAL AI ANALYSIS CONTEXT:
        The specialized DenseNet121 model has detected the following probabilities:
        {pred_context}
        
        TASK:
        1. Review the visual evidence in the X-Ray.
        2. Correlate with the AI findings provided above.
        3. Write a professional Radiology Report with 'CLINICAL OBSERVATIONS' and 'IMPRESSION' sections.
        4. Use clinical terminology (e.g., opacities, pleural spaces, cardiothoracic ratio).
        5. Keep the tone professional, objective, and cautious.
        
        Format the output clearly using Markdown.
        """

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        
        return message.content[0].text
