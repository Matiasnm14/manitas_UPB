import cv2
import mediapipe as mp
import torch

# Inicializamos utilidades de dibujo y el modelo de manos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def draw_landmarks(frame, results):
    """
    Dibuja los landmarks de las manos detectadas.
    'results' es el objeto retornado por hands.process()
    """
    if results.multi_hand_landmarks:
        # Iteramos sobre cada mano detectada (máximo 2 según la configuración)
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                # Estilos opcionales para que se vea más profesional
                # mp_drawing.get_default_hand_landmarks_style(),
                # mp_drawing.get_default_hand_connections_style()
            )


cap = cv2.VideoCapture(0)

# Configuramos el modelo de Hands
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    print("Iniciando detección... Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Espejo para que el movimiento sea natural
        frame = cv2.flip(frame, 1)

        # MediaPipe necesita RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen
        results = hands.process(rgb)

        # Dibujar resultados
        draw_landmarks(frame, results)

        cv2.imshow("MediaPipe Hands Only", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()