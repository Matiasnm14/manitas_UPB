import os
import cv2
import mediapipe as mp
import csv
import glob
import numpy as np


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


# 1. Configurar MediaPipe
# IMPORTANTE: Usamos static_image_mode=True porque son imágenes sueltas, no un video.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,  # Asumimos 1 mano por imagen en este dataset
    min_detection_confidence=0.5
)

# 2. Configurar rutas (¡CAMBIA ESTO POR LA RUTA DONDE DESCOMPRIMISTE EL DATASET!)
# Usaremos la carpeta de 'train' para este ejemplo
dataset_path = "./resources/train"
csv_path = "dataset_manos.csv"

# 3. Crear y preparar el archivo CSV
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Crear la primera fila (encabezados): 'label', 'x0', 'y0', 'z0', 'x1', 'y1'... hasta 20
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    writer.writerow(header)

    # 4. Buscar todas las imágenes .png en la carpeta
    image_files = glob.glob(os.path.join(dataset_path, "*.png"))
    print(f"Se encontraron {len(image_files)} imágenes. Procesando...")

    procesadas_exito = 0

    for i, img_path in enumerate(image_files):
        # Imprimir progreso cada 1000 imágenes
        if i % 1000 == 0 and i > 0:
            print(f"Procesadas {i}/{len(image_files)}...")

        # Leer la imagen
        img = cv2.imread(img_path)
        if img is None:
            continue

        # MediaPipe requiere formato RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe
        results = hands.process(img_rgb)

        # Si MediaPipe logró detectar una mano en la imagen...
        if results.multi_hand_landmarks:
            # Extraer la etiqueta del nombre del archivo (ej. "..._5L.png" -> "5")
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Separar por el guión bajo y tomar el primer carácter de la última parte
            label_str = name_without_ext.split('_')[-1][0]

            try:
                label = int(label_str)
            except ValueError:
                continue  # Saltar si hay algún archivo con nombre inesperado

            # Extraer los puntos y guardarlos
            for hand_landmarks in results.multi_hand_landmarks:
                row = [label]
                # Usamos nuestra nueva función
                landmarks_list = normalizar_landmarks(hand_landmarks)
                row.extend(landmarks_list)

                writer.writerow(row)
                procesadas_exito += 1

print("-" * 30)
print(f"¡Extracción terminada!")
print(f"MediaPipe detectó manos y guardó datos de {procesadas_exito} imágenes.")
print(f"Archivo guardado en: {csv_path}")