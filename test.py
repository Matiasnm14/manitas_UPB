import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from pyModbusTCP.client import ModbusClient  # <--- CAMBIO: Importamos Modbus en lugar de socket

# =========================================================
# CONFIGURACIÓN DE RED (MODBUS TCP HACIA FACTORY I/O)
# =========================================================
# Mantenemos la IP de la computadora donde está Factory I/O que nos pasaste
FACTORY_IO_IP = '172.16.72.199'
# ATENCIÓN: Modbus usa el puerto 502 por estándar (Factory I/O escucha ahí)
FACTORY_IO_PORT = 502
FACTORY_IO_PORT2 = 503


print(f"Conectando a Factory I/O en {FACTORY_IO_IP}:{FACTORY_IO_PORT}...")
# auto_open=True hace que el script intente reconectarse si la red falla
cliente_plc = ModbusClient(host=FACTORY_IO_IP, port=FACTORY_IO_PORT, auto_open=True)
cliente_plc2 = ModbusClient(host=FACTORY_IO_IP, port=FACTORY_IO_PORT2, auto_open=True)


if cliente_plc.open():
    print("¡Conectado a Factory I/O exitosamente!")
else:
    print("⚠️ No se pudo conectar a Factory I/O. Verifica que el Driver esté en 'Running'.")
    print("El programa de cámara seguirá funcionando sin enviar datos.")

if cliente_plc2.open():
    print("¡Conectado a Factory I/O exitosamente!")
else:
    print("⚠️ No se pudo conectar a Factory I/O. Verifica que el Driver esté en 'Running'.")
    print("El programa de cámara seguirá funcionando sin enviar datos.")
# =========================================================

def normalizar_landmarks(hand_landmarks):
    # Extraemos las coordenadas en una matriz de NumPy
    puntos = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # 1. Traslación: Hacemos que la muñeca (punto 0) sea el centro (0,0,0)
    base = puntos[0]
    puntos_centrados = puntos - base

    # 2. Escala: Normalizamos el tamaño de la mano
    # Buscamos la distancia máxima para que todos los valores queden entre -1 y 1
    valor_maximo = np.max(np.abs(puntos_centrados))
    if valor_maximo > 0:
        puntos_normalizados = puntos_centrados / valor_maximo
    else:
        puntos_normalizados = puntos_centrados

    # Devolvemos una lista plana de 63 elementos lista para PyTorch
    return puntos_normalizados.flatten().tolist()


# 1. Configuración de Dispositivo (GPU CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trabajando con: {device}")


# 2. Definimos la MISMA arquitectura que usamos para entrenar
class DetectorDeDedos(nn.Module):
    def __init__(self):
        super(DetectorDeDedos, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6 clases (0 a 5 dedos)
        )

    def forward(self, x):
        return self.red(x)


# 3. Instanciamos el modelo, cargamos el "cerebro" y lo pasamos a la GPU
modelo = DetectorDeDedos()
# Usamos weights_only=True por seguridad al cargar (buena práctica en PyTorch)
modelo.load_state_dict(torch.load("modelo_dedos.pth", map_location=device, weights_only=True))
modelo.to(device)
modelo.eval()  # Modo evaluación (apaga Dropout/BatchNorm si los hubiera, optimiza inferencia)
print("Modelo cargado exitosamente.")

# Inicializamos utilidades de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


cap = cv2.VideoCapture(0)

# Configuramos MediaPipe
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=8,  # Tu código permite hasta 8 manos
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as hands:
    print("Iniciando... Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        alto, ancho, _ = frame.shape

        # Procesamiento de MediaPipe (CPU)
        results = hands.process(rgb)

        # --- SECCIÓN TORCH + CUDA ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraemos los puntos NORMALIZADOS
                landmarks_list = normalizar_landmarks(hand_landmarks)

                # Convertimos a Tensor y enviamos a la GPU
                input_tensor = torch.tensor([landmarks_list], dtype=torch.float32).to(device)

                # INFERENCIA: Hacemos la predicción
                with torch.no_grad():  # Evitamos que la GPU guarde gradientes
                    output = modelo(input_tensor)  # Esto devuelve 6 probabilidades
                    prediccion = torch.argmax(output, dim=1).item()  # Tomamos el índice de la mayor probabilidad

                # =========================================================
                # CONTROL DE FACTORY I/O (MODBUS)
                # =========================================================
                if cliente_plc.is_open:
                    # Lógica: 5 dedos enciende la banda (Coil 0), 0 dedos (puño) la apaga
                    if prediccion == 5:
                        cliente_plc.write_single_coil(0, True)
                        cliente_plc2.write_single_coil(0, True)
                        cv2.putText(frame, "CINTA: ON", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    elif prediccion == 0:
                        cliente_plc.write_single_coil(0, False)
                        cliente_plc2.write_single_coil(0, False)
                        cv2.putText(frame, "CINTA: OFF", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # =========================================================

                # Dibujamos el resultado en la pantalla
                cx = int(hand_landmarks.landmark[0].x * ancho)
                cy = int(hand_landmarks.landmark[0].y * alto)

                # Ponemos el texto de la predicción justo arriba de la muñeca
                cv2.putText(frame, f"Dedos: {prediccion}", (cx - 20, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Dibujamos las líneas y puntos de las manos
        draw_landmarks(frame, results)

        cv2.imshow("MediaPipe + PyTorch + Factory I/O", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Apagamos el motor por seguridad al cerrar el programa
            if cliente_plc.is_open:
                cliente_plc.write_single_coil(0, False)
            break

cap.release()
cv2.destroyAllWindows()