import cv2
import mediapipe as mp
import numpy as np
import joblib

# Carregando o modelo
modelo = joblib.load('modelo.pkl')

# Inicializando o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Inicializando o MediaPipe DrawingUtils
mp_drawing = mp.solutions.drawing_utils

# Caminho para o vídeo de teste
caminho_video = './inputs/20210123070155_600c9cd37e0ac.mp4'

# Lendo o vídeo
cap = cv2.VideoCapture(caminho_video)

# Lista para armazenar as coordenadas das mãos
coordenadas = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertendo a cor da imagem
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
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
            
            # Adicionando a previsão como uma legenda na imagem
            cv2.putText(image, f'Previsão: {previsao_frame[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Convertendo a cor da imagem de volta para BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Exibindo a imagem
    cv2.imshow('Video', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Fazendo a média das coordenadas
coordenadas_media = np.mean(coordenadas, axis=0)

# Fazendo a previsão para o vídeo completo
previsao_video = modelo.predict([coordenadas_media])

print(f'O vídeo foi previsto como: {previsao_video}')
