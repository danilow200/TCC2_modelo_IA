import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image

# Carregando os modelos
modelo_uma = joblib.load('modelo_uma_mao.pkl')
modelo_duas = joblib.load('modelo_duas_maos.pkl')

# Inicializando o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Inicializando o MediaPipe DrawingUtils
mp_drawing = mp.solutions.drawing_utils

# Inicializando a webcam
cap = cv2.VideoCapture(0)

# Perguntando qual é a mão dominante
mao_dominante = input("Qual é a mão dominante? (direita/esquerda): ").strip().lower()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertendo a cor da imagem
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    
    # Processando a imagem
    result = hands.process(image)
    
    if result.multi_hand_landmarks:
        if len(result.multi_hand_landmarks) == 1:
            # Apenas uma mão detectada
            hand_landmarks = result.multi_hand_landmarks[0]
            handedness = result.multi_handedness[0]

            # Extraindo as coordenadas
            coords = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            
            # Verificando se a mão detectada é a mão direita ou esquerda
            if handedness.classification[0].label.lower() == 'right':
                hand_label = 'direita'
            else:
                hand_label = 'esquerda'
            
            # Invertendo as coordenadas se a mão dominante for a esquerda e a mão detectada for a direita
            if mao_dominante == 'esquerda' and hand_label == 'direita':
                coords = np.array([[1 - landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            
            # Fazendo a previsão com o modelo para uma mão
            previsao_frame = modelo_uma.predict([coords])
            
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
        
        elif len(result.multi_hand_landmarks) == 2:
            # Duas mãos detectadas
            coords_direita = None
            coords_esquerda = None
            
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
            
            if coords_direita is not None and coords_esquerda is not None:
                # Fazendo a previsão com o modelo para duas mãos (considerando as duas mãos)
                previsao_frame_direita = modelo_duas.predict([coords_direita])
                previsao_frame_esquerda = modelo_duas.predict([coords_esquerda])
                
                # Desenhando as landmarks na imagem
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Convertendo a imagem do cv2 para PIL
                pil_image = Image.fromarray(image)
                
                # Criando um objeto de desenho
                draw = ImageDraw.Draw(pil_image)
                
                # Escolhendo a fonte e o tamanho
                font = ImageFont.truetype("arial.ttf", 30)

                # Desenhando o texto para ambas as mãos
                draw.text((10, 30), f'Previsão Direita: {previsao_frame_direita[0]}', font=font, fill=(0, 255, 0, 0))
                draw.text((10, 70), f'Previsão Esquerda: {previsao_frame_esquerda[0]}', font=font, fill=(0, 255, 0, 0))
                
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
