# ☁️ Deployment Guide — ChestX-Ray AI Dashboard

> Step-by-step instructions for deploying the dashboard to multiple cloud platforms.

---

## 🚀 Option 1: Streamlit Cloud (Free, Fastest — 3 Minutes)

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "feat: v1.0 production release"
   git tag v1.0
   git push origin main --tags
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. **Click "New App"** → Select your repo → Set the main file path to:
   ```
   dashboard/app.py
   ```

4. **Add Secrets** (Streamlit Cloud Secrets Manager):
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ROBOFLOW_API_KEY = "your_key_here"
   ```

5. **Click Deploy** → Your app will be live at:
   ```
   https://your-username-chestxray-ai.streamlit.app
   ```

---

## 🤗 Option 2: Hugging Face Spaces (Free, ML Community Visibility)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create New Space**.
2. Select **Streamlit** as the SDK.
3. Upload your project files or connect your GitHub repo.
4. Add secrets via the Space Settings → **Repository Secrets**.

---

## 🐳 Option 3: Docker on a Cloud VM (Most Control)

```bash
# On your server (e.g., AWS EC2, Google Cloud Run, Azure):
git clone https://github.com/yourusername/chestxray-ai.git
cd chestxray-ai
cp .env.example .env       # Add your API keys
make docker-run            # One command to launch
```

The app will be available at `http://your-server-ip:8501`.

---

## 🔒 Security Checklist Before Going Live

- [ ] `.env` is in `.gitignore` (never push API keys to GitHub)
- [ ] `enableXsrfProtection = true` is set in `.streamlit/config.toml`
- [ ] All patient/test images are anonymized before uploading
- [ ] Clinical disclaimer is visible on the dashboard

---

## 📊 Post-Deployment: Validation Steps

1. Upload a test X-ray from `Classification-5/test/` folder.
2. Verify Gemini report is generated in < 30 seconds.
3. Verify Roboflow bounding boxes appear correctly.
4. Run `make health-check` to confirm all APIs are responding.

---

> ⚠️ **Disclaimer**: This system is for research and portfolio demonstration purposes only. Not intended for clinical use without licensed radiologist oversight.
