import os
import requests
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

class RoboflowDetector:
    """
    Handles object detection using Roboflow's REST API.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ROBOFLOW_API_KEY", "eTrSU1WAJfSUcSCttQQ4")
        self.model_id = "train-3jix3/4"
        self.url = f"https://detect.roboflow.com/{self.model_id}?api_key={self.api_key}"

    def detect(self, image_bytes, custom_url=None):
        """
        Sends image bytes to Roboflow detection endpoint.
        """
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        target_url = custom_url if custom_url else self.url

        try:
            response = requests.post(
                target_url,
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=20
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Roboflow HTTP {response.status_code}: {response.text[:300]}")
        except requests.exceptions.Timeout:
            raise Exception("Roboflow API timed out (20s). Check your internet connection.")
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"Cannot reach Roboflow API. Check internet. Detail: {str(e)[:100]}")
        except Exception as e:
            raise Exception(f"Roboflow error: {str(e)}")

    def classify_covid(self, image_bytes):
        """
        Calls the specialized COVID-19 classification model.
        """
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        # Ensure we are using the correct classification workspace/model
        url = f"https://classify.roboflow.com/classification-upura/5?api_key={self.api_key}"
        
        try:
            response = requests.post(
                url,
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=20
            )
            if response.status_code == 200:
                data = response.json()
                # Debug print for internal logs
                if 'predictions' not in data:
                    raise Exception(f"Unexpected API response format: {data}")
                return data
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
        except Exception as e:
            raise Exception(f"{str(e)}")


    def draw_boxes(self, image, predictions):
        """
        Draws bounding boxes on the image.
        """
        draw = ImageDraw.Draw(image)
        # Try to load a font, otherwise use default
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for pred in predictions.get('predictions', []):
            if 'x' not in pred: continue # Skip classification results
            
            x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            label = pred['class']
            conf = pred['confidence']
            
            # Calculate box corners
            left = x - w/2
            top = y - h/2
            right = x + w/2
            bottom = y + h/2
            
            # Draw box
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            # Draw label
            draw.text((left, top - 25), f"{label} {conf:.1%}", fill="red", font=font)
            
        return image
