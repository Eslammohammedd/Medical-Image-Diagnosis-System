import torch
import os
from models.architectures.densenet121 import get_model
from training.transforms import get_transforms
from PIL import Image
import numpy as np

class ChestXrayPredictor:
    """
    Main interface for running local inference and explainability.
    """
    def __init__(self, model_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 5
        self.classes = [
            "Bacterial Pneumonia", "Covid-19", "Normal", "Pneumonia", "Pneumonia Viral"
        ]
        
        # Initialize model
        self.model = get_model(num_classes=self.num_classes, pretrained=False)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded model weights from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load weights: {e}. Using random initialization.")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_transforms(is_train=False)

    def predict(self, image_path):
        """
        Runs prediction and returns probability scores.
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        results = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return results

    def get_top_class(self, results):
        return max(results, key=results.get)
