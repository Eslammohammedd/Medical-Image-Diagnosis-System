#!/usr/bin/env python
"""
scripts/setup_env.py
─────────────────────
One-shot setup script.  Run once after cloning the repo:

    python scripts/setup_env.py

What it does:
  1. Checks Python version (>= 3.9)
  2. Creates required local directories
  3. Copies .env.example → .env (if .env doesn't exist)
  4. Prints next-step instructions
"""
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python():
    version = sys.version_info
    if version < (3, 9):
        print(f"❌ Python {version.major}.{version.minor} detected. Python >= 3.9 required.")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def create_directories():
    dirs = [
        "models/weights",
        "logs",
        "dashboard/assets",
        "data/raw",
        "data/processed",
    ]
    for d in dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {d}/")


def setup_env():
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"
    if env_file.exists():
        print("✅ .env already exists — skipping copy.")
    elif env_example.exists():
        shutil.copy(env_example, env_file)
        print("📄 .env created from .env.example")
        print("   ⚠️  Remember to fill in your API keys in .env before running the app!")
    else:
        print("⚠️  .env.example not found. Create .env manually.")


def print_next_steps():
    print("\n" + "=" * 55)
    print("🚀 Setup complete!  Next steps:")
    print("=" * 55)
    print("1. Edit .env and add your API keys:")
    print("     GEMINI_API_KEY=...")
    print("     ROBOFLOW_API_KEY=...")
    print()
    print("2. Activate the virtual environment:")
    print("     venv_gpu\\Scripts\\activate   (Windows)")
    print("     source venv_gpu/bin/activate (Linux/Mac)")
    print()
    print("3. Install dependencies:")
    print("     pip install -r requirements.txt")
    print()
    print("4. Run the dashboard:")
    print("     streamlit run dashboard/app.py")
    print()
    print("5. (Optional) Train the local model:")
    print("     python training/train.py --config configs/stage1_chexpert.yaml")
    print("=" * 55)


if __name__ == "__main__":
    print("\n🏥 ChestX-Ray AI — Environment Setup\n")
    check_python()
    create_directories()
    setup_env()
    print_next_steps()
