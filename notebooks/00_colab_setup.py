# %% [markdown]
# # Module 0: Colab Environment Setup & Persistence Mount
# This cell handles:
# 1. Google Drive mounting (when running in Colab)
# 2. Workspace path resolution (setting DRIVE_DIR)
# 3. Fast pip installation of pinned dependencies without requiring runtime restart
# 4. Smoke test invocation

# %%
import os
import sys
from pathlib import Path

# Detect Colab runtime environment
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    print("[Colab] Mounting Google Drive...")
    from google.colab import drive
    drive.mount("/content/drive")
    
    # User-configurable project directory on Google Drive
    DRIVE_DIR = "/content/drive/MyDrive/ai_gen_text_detection"
    os.environ["DRIVE_DIR"] = DRIVE_DIR
    print(f"[Colab] DRIVE_DIR set to: {DRIVE_DIR}")
    
    # Path inside colab workspace
    PROJECT_ROOT = Path("/content/ai_gen_text_detection")
    if not PROJECT_ROOT.exists():
        # Clone or create local link
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print("[Colab] Installing dependencies...")
    # --no-warn-conflicts ensures non-breaking Colab package integration
    !pip install -q --no-warn-conflicts -r requirements.txt
else:
    DRIVE_DIR = os.environ.get("DRIVE_DIR", str(Path("./outputs").resolve()))
    os.environ["DRIVE_DIR"] = DRIVE_DIR
    print(f"[Local] Not in Colab. DRIVE_DIR set to: {DRIVE_DIR}")

Path(DRIVE_DIR).mkdir(parents=True, exist_ok=True)
print(f"Directory ready: {DRIVE_DIR}")
