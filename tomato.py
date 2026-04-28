import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Все важные признаки
CONFIG = {
    "data_path": r"C:\Users\RobotComp.ru\.cache\kagglehub\datasets\abdallahalidev\plantvillage-dataset\versions\3\plantvillage dataset\color\tomato",
    "img_size": (150, 150),
    "batch_size": 32,
    "seed": 42,
    "validation_split": 0.1,
    "epochs": 20,
    "learning_rate": 0.0001
}

#Загружаем тренировочные данные
raw_train = keras.utils.image_dataset_from_directory(
    CONFIG["data_path"],
    image_size=CONFIG["img_size"],
    batch_size=CONFIG["batch_size"],
    seed=CONFIG["seed"],
    subset="training",
    validation_split=CONFIG["validation_split"])

#Загружаем тестовые данные
raw_val = keras.utils.image_dataset_from_directory(
    CONFIG["data_path"],
    image_size=CONFIG["img_size"],
    batch_size=CONFIG["batch_size"],
    seed=CONFIG["seed"],
    subset="validation",
    validation_split=CONFIG["validation_split"])

#Нормализируем
train_ds = raw_train.map(lambda img, label: (img / 255.0, label))
val_ds = raw_val.map(lambda img, label: (img / 255.0, label))

#Аргументация
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.1),                   
    keras.layers.RandomZoom(0.1),               
])

#Собираем модель
model = keras.Sequential([
    data_augmentation,
    keras.layers.Conv2D(32, 3, activation="relu", input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128, 3, activation="relu"),
    keras.layers.MaxPooling2D(2,2),


    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(raw_train.class_names), activation="softmax"),
    ])

#early_stop - для предотвращения обучения
early_stop = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True)

optimizer = keras.optimizers.Adam(CONFIG["learning_rate"])#Выбираем Adam как оптимизацию
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])#Компилируем модель
history = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[early_stop])#Обучаем

model.save("tomato.keras")#Запускаем модель

#Строим график
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.title("График обучения")

plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend() 
plt.grid(True)
plt.show()
