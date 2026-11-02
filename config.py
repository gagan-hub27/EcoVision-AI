import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT=Path(__file__).resolve().parent
APP_NAME="EcoVision AI"
MODEL_PATH=os.getenv("MODEL_PATH",str(ROOT/"models"/"best.pt"))
CONFIDENCE_THRESHOLD=float(os.getenv("CONFIDENCE_THRESHOLD","0.35"))
MONGODB_URI=os.getenv("MONGODB_URI","").strip()
MONGODB_DATABASE=os.getenv("MONGODB_DATABASE","ecovision_ai")
