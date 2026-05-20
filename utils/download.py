import os
import subprocess
import sys

def download_datasets():
    """
    Downloads the essential datasets for the Chest X-Ray project using Kaggle API.
    Uses the absolute path for kaggle.exe on this system.
    """
    kaggle_path = r"C:\Users\eslam\AppData\Roaming\Python\Python314\Scripts\kaggle.exe"
    
    datasets = {
        "nih-chest-xrays": "nih-chest-xrays/data",
        "covid-19-radiography": "tawsifurrahman/covid19-radiography-database",
        "vinbigdata-cxr": "vinbigdata-chest-xray-abnormalities-detection"
    }
    
    # Check if kaggle exists
    if not os.path.exists(kaggle_path):
        print(f"ERROR: Kaggle executable not found at {kaggle_path}")
        return

    # 1. Download COVID-19 Dataset (Verification)
    print("Starting download: COVID-19 Radiography Database...")
    covid_path = "data/raw/covid"
    if not os.path.exists(covid_path):
        os.makedirs(covid_path)
        
    try:
        subprocess.run([
            kaggle_path, "datasets", "download", "-d", datasets["covid-19-radiography"], 
            "-p", covid_path, "--unzip"
        ], check=True)
        print("SUCCESS: COVID-19 Dataset downloaded.")
    except Exception as e:
        print(f"FAILED: COVID-19 download failed: {e}")

    # 2. Download NIH Dataset (Large)
    print("\nStarting download: NIH ChestX-ray14 (approx 42GB)...")
    nih_path = "data/raw/nih"
    if not os.path.exists(nih_path):
        os.makedirs(nih_path)
    
    try:
        subprocess.run([
            kaggle_path, "datasets", "download", "-d", datasets["nih-chest-xrays"], 
            "-p", nih_path, "--unzip"
        ], check=True)
        print("SUCCESS: NIH Dataset downloaded and unzipped.")
    except Exception as e:
        print(f"FAILED: NIH download failed: {e}")

if __name__ == "__main__":
    download_datasets()
