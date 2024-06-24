import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image

# Carregando o modelo
modelo = joblib.load('modelo.pkl')

# Inicializando o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Inicializando o MediaPipe DrawingUtils
mp_drawing = mp.solutions.drawing_utils

# Inicializando a webcam
cap = cv2.VideoCapture(0)

# Lista para armazenar as coordenadas das mãos
coordenadas = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertendo a cor da imagem
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image,1)
    
    # Processando a imagem
    result = hands.process(image)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extraindo as coordenadas
            coords = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten())
            
            # Adicionando as coordenadas à lista
            coordenadas.append(coords)
            
            # Fazendo a previsão para o frame atual
            previsao_frame = modelo.predict([coords])
            
            # Desenhando as landmarks na imagem
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Convertendo a imagem do cv2 para PIL
            pil_image = Image.fromarray(image)
            
            # Criando um objeto de desenho
            draw = ImageDraw.Draw(pil_image)
            
            # Escolhendo a fonte e o tamanho
            font = ImageFont.truetype("arial.ttf", 30)
            
            # Desenhando o texto
            draw.text((10, 30), f'Previsão: {previsao_frame[0]}', font=font, fill=(0, 255, 0, 0))
            
        # Convertendo a imagem de volta para cv2
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Exibindo a imagem
    cv2.imshow('Video', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
