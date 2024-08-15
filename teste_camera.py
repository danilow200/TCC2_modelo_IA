import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image

# Carregando o modelo e os scalers
modelo = joblib.load('modelo.pkl')
scaler_direita = joblib.load('scaler_direita.pkl')
scaler_esquerda = joblib.load('scaler_esquerda.pkl')

# Inicializando o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Inicializando o MediaPipe DrawingUtils
mp_drawing = mp.solutions.drawing_utils

# Inicializando a webcam
cap = cv2.VideoCapture(0)

# Perguntando qual é a mão dominante
mao_dominante = input("Qual é a mão dominante? (direita/esquerda): ").strip().lower()

anterior = 'Não identificado'
previsoes_ultimos_frames = []
previsao_mais_comum = anterior  # Inicializa com a previsão anterior

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertendo a cor da imagem
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    
    # Processando a imagem
    result = hands.process(image)
    
    coords_direita = None
    coords_esquerda = None

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Extraindo as coordenadas
            coords = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            
            # Verificando se a mão detectada é a mão direita ou esquerda
            if handedness.classification[0].label.lower() == 'right':
                hand_label = 'direita'
                coords_direita = coords
            else:
                hand_label = 'esquerda'
                coords_esquerda = coords
            
            # Invertendo as coordenadas se a mão dominante for a esquerda e a mão detectada for a direita
            if mao_dominante == 'esquerda' and hand_label == 'direita':
                coords_direita = np.array([[1 - landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            elif mao_dominante == 'direita' and hand_label == 'esquerda':
                coords_esquerda = np.array([[1 - landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            
            # Desenhando as landmarks na imagem
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Normalizar e concatenar as coordenadas, considerando se uma ou ambas as mãos forem detectadas
        if coords_direita is not None and coords_esquerda is not None:
            coords_direita = scaler_direita.transform([coords_direita])[0]
            coords_esquerda = scaler_esquerda.transform([coords_esquerda])[0]
            coords_combined = np.concatenate((coords_direita, coords_esquerda))
        elif coords_direita is not None:
            coords_direita = scaler_direita.transform([coords_direita])[0]
            coords_combined = np.concatenate((coords_direita, np.zeros_like(coords_direita)))
        elif coords_esquerda is not None:
            coords_esquerda = scaler_esquerda.transform([coords_esquerda])[0]
            coords_combined = np.concatenate((np.zeros_like(coords_esquerda), coords_esquerda))
        else:
            coords_combined = None
        
        if coords_combined is not None:
            # Fazendo a previsão para o frame atual
            previsao_frame = modelo.predict([coords_combined])[0]
            previsoes_ultimos_frames.append(previsao_frame)
            
            # Manter apenas as últimas 5 previsões
            if len(previsoes_ultimos_frames) > 5:
                previsoes_ultimos_frames.pop(0)
            
            # Determinar a previsão mais comum entre os últimos 5 frames
            previsao_mais_comum = max(set(previsoes_ultimos_frames), key=previsoes_ultimos_frames.count)
            
            anterior = previsao_mais_comum
            
    # Convertendo a imagem do cv2 para PIL
    pil_image = Image.fromarray(image)
    
    # Criando um objeto de desenho
    draw = ImageDraw.Draw(pil_image)
    
    # Escolhendo a fonte e o tamanho
    font = ImageFont.truetype("arial.ttf", 30)
    
    # Desenhando o texto
    draw.text((10, 30), f'Previsão: {previsao_mais_comum}', font=font, fill=(255, 255, 255, 0))
    
    # Convertendo a imagem de volta para cv2
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Exibindo a imagem
    cv2.imshow('Video', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
