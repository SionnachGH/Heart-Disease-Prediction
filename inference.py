import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Загрузка тестовых данных
test = pd.read_csv("test.csv")
X_test = test.drop(columns=["ID"])

# Масштабирование (на практике scaler нужно сохранять, здесь переобучаем для простоты)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Архитектура модели (должна быть точно такой же, как при обучении)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_test_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Загрузка весов
model.load_weights("best_model.weights.h5")

# Предсказания
preds = model.predict(X_test_scaled).ravel()
labels = (preds > 0.5).astype(int)

# Сохранение сабмишена
submission = pd.DataFrame({
    "ID": test["ID"],
    "class": labels
})
submission.to_csv("submission.csv", index=False)
print("Готово: сохранён файл submission.csv")
