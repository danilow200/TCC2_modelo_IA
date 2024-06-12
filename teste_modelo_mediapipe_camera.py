import numpy as np
import cv2
import mediapipe as mp
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter

# Carregar os marcos e as classes
landmarks_train = np.load('landmarks.npy')
classes = np.load('classes_mediapipe.npy')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Função para processar um quadro do vídeo
def process_frame(frame):
    # Converter a cor da imagem de BGR para RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem e obter os resultados da detecção da mão
    results = hands.process(rgb_image)

    landmarks = []
    if results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) == 2:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenhar os pontos de referência da mão na imagem
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Salvar as coordenadas dos pontos de referência da mão
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])
    else:
        landmarks = [[0, 0, 0] for _ in range(42)]  # Supondo que haja 21 pontos de referência por mão

    return frame, landmarks

# Função para fazer previsões em tempo real
def predict_on_camera(landmarks_train, classes):
    cap = cv2.VideoCapture(0)  # Usar a câmera principal
    predictions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, landmarks_test = process_frame(frame)
        if landmarks_test is not None:
            # Achatar os arrays de marcos
            landmarks_test_flat = np.array(landmarks_test).flatten()
            landmarks_train_flat = np.array(landmarks_train).reshape(len(landmarks_train), -1)
            # Calcular a distância euclidiana entre os marcos de teste e treinamento
            distances = euclidean_distances([landmarks_test_flat], landmarks_train_flat)
            # Encontrar a classe correspondente ao menor distância
            min_distance_index = np.argmin(distances)
            if min_distance_index < len(classes):
                predicted_class = classes[min_distance_index]
                predictions.append(predicted_class)
                # Exibir a previsão na tela
                cv2.putText(frame, f"Previsão: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Exibir o quadro
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Encontrar a classe mais frequente nas previsões
    most_common_class = Counter(predictions).most_common(1)[0][0]
    print(f"Previsão mais frequente: {most_common_class}")

# Testar em tempo real
predict_on_camera(landmarks_train, classes)
