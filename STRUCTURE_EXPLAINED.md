# 🗺️ Technical Architecture & Directory Map

This document explains the full structure of the **ChestX-Ray AI Dashboard** and how all components are interconnected.

---

## 📂 Directory Breakdown

### 1. `dashboard/` (The Face)
- **`app.py`**: The main entry point. It handles the UI, file uploads, and coordinates between the different AI models.
- **`.streamlit/config.toml`**: Optimized production settings for the UI.

### 2. `inference/` (The Brains)
- **`gemini_integration.py`**: Connects to Google Gemini 1.5 Flash for report generation.
- **`roboflow_integration.py`**: Handles both Object Detection (locations) and specialized COVID classification via Roboflow API.
- **`predictor.py`**: Loads the local PyTorch model (`DenseNet121`) and performs multi-label classification.
- **`gradcam.py`**: Generates visual heatmaps to explain the local model's focus.

### 3. `models/` (The Architectures)
- **`architectures/`**: Contains Python definitions for `densenet121.py`, `resnet50.py`, etc.
- **`weights/`**: Where the trained `.pth` files are stored.

### 4. `training/` (The Factory)
- **`train.py`**: The main training loop.
- **`dataset.py`**: Logic for reading real medical images and labels.
- **`trainer.py`**: Handles the optimization, loss calculation, and checkpoint saving.

### 5. `utils/` (The Tools)
- **`health_check.py`**: A diagnostic script to verify system readiness.
- **`download.py`**: Automated script to fetch datasets from Kaggle/Roboflow.

---

## 🔄 How Files are Interconnected (Data Flow)

1. **Initialization**: `app.py` loads settings from `.env` via `load_dotenv()`.
2. **Input**: User uploads an image in `app.py`.
3. **Local Analysis**: 
   - `app.py` calls `ChestXrayPredictor` from `inference/predictor.py`.
   - `predictor.py` uses architectures from `models/architectures/`.
4. **Spatial Analysis**:
   - `app.py` calls `RoboflowDetector` from `inference/roboflow_integration.py` to get bounding boxes.
5. **Synthesis**:
   - `app.py` sends the image + local results to `GeminiRadiologyReporter` in `inference/gemini_integration.py`.
6. **Output**: The final UI combines the heatmap, boxes, and the Gemini report.

---

## 🛠️ Automation & Deployment Links
- **`Makefile`**: Bridges the CLI to the code. Running `make run` executes `streamlit run dashboard/app.py`.
- **`Dockerfile`**: Bridges the code to the OS. It installs `requirements.txt` and sets up the environment to run the `CMD`.
- **`docker-compose.yml`**: Bridges the container to your hardware (mapping ports and mounting data folders).

---

## 🛡️ Security & Config
- **`.env`**: Centralized storage for API keys used by all integration scripts.
- **`requirements.txt`**: Pinned versions ensure that an update in one library doesn't break the connection between others.

**Project Status: Fully Integrated & Synchronized.** ⚙️💎🦾
