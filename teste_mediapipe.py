import cv2
import mediapipe as mp
import numpy as np
import joblib

# Carregando o modelo e os scalers
modelo = joblib.load('modelo.pkl')
scaler_direita = joblib.load('scaler_direita.pkl')
scaler_esquerda = joblib.load('scaler_esquerda.pkl')

# Inicializando o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Inicializando o MediaPipe DrawingUtils
mp_drawing = mp.solutions.drawing_utils

# Caminho para o vídeo de teste
caminho_video = './inputs/20210123070155_600c9cd37e0ac.mp4'

# Lendo o vídeo
cap = cv2.VideoCapture(caminho_video)

# Lista para armazenar as previsões dos frames
previsoes_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertendo a cor da imagem
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
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
                coords_direita = coords
            else:
                coords_esquerda = coords
            
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
            previsoes_frames.append(previsao_frame)
            
            # Adicionando a previsão como uma legenda na imagem
            cv2.putText(image, f'Previsão: {previsao_frame}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Convertendo a cor da imagem de volta para BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Exibindo a imagem
    cv2.imshow('Video', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Fazendo a média das previsões dos frames
previsao_mais_comum = max(set(previsoes_frames), key=previsoes_frames.count)

print(f'O vídeo foi previsto como: {previsao_mais_comum}')
