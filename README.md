# Tomato 🍅 Leaf Disease Diagnosis Web Service (CV / Deep Learning)

A practical pet project that recognizes 10 classes of tomato leaf states and diseases based on the PlantVillage dataset.

## Tech Stack ⚡

- **Backend:** Python, FastAPI, Uvicorn
- **ML/DL:** TensorFlow, Keras, NumPy, PIL (Pillow)
- **Frontend:** HTML5, CSS3, JavaScript (asynchronous requests via Fetch API)

## Project Structure 👁️ 

- `tomato.py` — FastAPI backend server responsible for image uploading, matrix preprocessing in NumPy (normalization via /255.0 and dimension expansion for batching), and model inference.
- `alert.html` — A clean frontend interface for user photo uploads.
- `load.py` — A script for local verification or data loading.
