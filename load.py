from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Загрузка модели
model = tf.keras.models.load_model("tomato.keras")

# Список классов (10 болезней + здоровый)
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_yellow_Leaf_Curl_Virus",
]
# FastAPI
app = FastAPI()

@app.get("/")
def root():
    # Возвращаем HTML‑страницу 
    return FileResponse("alert.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Читаем файл
    contents = await file.read()
    # Открываем, ресайзим
    img = Image.open(io.BytesIO(contents)).resize((150, 150))
    # Нормализуем и добавляем размерность батча
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Предсказание
    pred_probs = model.predict(img_array)[0]   # массив вероятностей (10)
    predicted_idx = np.argmax(pred_probs)       # индекс класса
    confidence = float(pred_probs[predicted_idx])
    
    return {
        "класс": CLASS_NAMES[predicted_idx],
        "уверенность": f"{confidence:.4f}"
    }
