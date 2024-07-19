import cv2
import numpy as np
import time
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Carregar o modelo treinado
model = load_model('libras_model.h5')

# Codificar rótulos de classe como inteiros
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy')

# Abrir a webcam
cap = cv2.VideoCapture(0)

# Dicionário para acompanhar as ocorrências de cada previsão
predictions = {}

while cap.isOpened():
    # Ler o próximo quadro
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Converter o quadro para escala de cinza e redimensionar para o tamanho de entrada do modelo
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))

    # Pré-processar a imagem
    img = img.reshape(-1, 64, 64, 1)
    img = img.astype('float32') / 255.0

    # Fazer a previsão
    pred = model.predict(img)
    class_index = np.argmax(pred, axis=1)

    # Decodificar a previsão para obter o rótulo da classe
    class_label = encoder.inverse_transform(class_index)[0]

    # Atualizar o dicionário de previsões
    if class_label in predictions:
        predictions[class_label] += 1
    else:
        predictions[class_label] = 1

    print(f"Previsão: {class_label}")

    # time.sleep(3)

cap.release()
cv2.destroyAllWindows()

# Encontrar a previsão com a maior ocorrência
max_prediction = max(predictions, key=predictions.get)
print(f"A previsão com a maior ocorrência foi: {max_prediction}")
