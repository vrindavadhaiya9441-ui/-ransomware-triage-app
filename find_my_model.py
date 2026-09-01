"""
find_my_model.py — run this IN GOOGLE COLAB to see what trained artefacts
you already have, and copy the ones this app needs into a local models/ folder.

You said you're not sure what you saved — this tells you in ten seconds.

USAGE (paste into a Colab cell):
    from google.colab import drive; drive.mount('/content/drive')
    %run find_my_model.py
"""
import os, glob, shutil

PROJECT_DIR = "/content/drive/MyDrive/ransomware-triage"   # <-- your project path
OUT = "/content/triage_artifacts"                          # local copy target
os.makedirs(OUT, exist_ok=True)

print(f"Scanning: {PROJECT_DIR}\n" + "-" * 60)
if not os.path.isdir(PROJECT_DIR):
    print("Project folder not found. Edit PROJECT_DIR at the top of this file.")
else:
    hits = []
    for ext in ("*.joblib", "*.pkl", "*.json", "*.parquet", "*.csv"):
        for p in glob.glob(os.path.join(PROJECT_DIR, "**", ext), recursive=True):
            hits.append(p)
    if not hits:
        print("No model-like files found under the project folder.")
    for p in sorted(hits):
        size = os.path.getsize(p) / 1024
        print(f"  {size:8.1f} KB  {p}")

    # Anything that looks like a model or explainer gets copied out for you
    print("\nCopying likely artefacts to", OUT)
    for p in sorted(hits):
        base = os.path.basename(p).lower()
        if any(k in base for k in ("model", "explainer", "shap", "feature",
                                   "threshold", "calibrat")):
            shutil.copy(p, os.path.join(OUT, os.path.basename(p)))
            print("  copied", os.path.basename(p))

print("\nNext: download the files in", OUT,
      "and drop them into this app's  models/  folder.")
print("If nothing useful was found, just put your MLRan feature CSV in  data/  "
      "and the app will train a fresh model on first run.")
