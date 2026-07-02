# импорт библиотек
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Загрузка модели
model = tf.keras.models.load_model("tomato.keras")

# Список классов (9 болезней + здоровый)
CLASS_NAMES = ['Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy']

app = FastAPI()


@app.get("/")# главный файл
def root():
    return FileResponse("alert.html")


@app.post("/predict")# серверная часть
async def predict(file: UploadFile = File(...)):
    read = await file.read()# получаем файл в виде бинарных чисел
    img = Image.open(io.BytesIO(read)).resize((150, 150))# загружаем в оперативную память, чтобы получиться array(массив)
    img_array = np.array(img) / 255.0# нормализуем
    img_array = np.expand_dims(img_array, axis=0)# размерность

    prediction = model.predict(img_array)[0]# получаем 1 вероятность
    prediction_index = np.argmax(prediction)# класс

    confidence = float(prediction[prediction_index])# строка с уверенностью

    return {
        "класс": CLASS_NAMES[prediction_index], # название болезни из списка
        "уверенность": f"{confidence * 100:.2f}%" # переводим вероятность в проценты
    }
