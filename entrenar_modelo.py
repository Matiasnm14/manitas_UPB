import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np



# 1. Configurar la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")


# 2. Crear la clase Dataset para leer el CSV
class DatasetManos(Dataset):
    def __init__(self, csv_file):
        # Leemos el CSV
        self.data = pd.read_csv(csv_file)

        # La columna 0 es la etiqueta (label). Las columnas 1 a 63 son las coordenadas.
        self.x = self.data.iloc[:, 1:].values.astype(np.float32)
        self.y = self.data.iloc[:, 0].values.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 3. Definir la Red Neuronal
class DetectorDeDedos(nn.Module):
    def __init__(self):
        super(DetectorDeDedos, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 6 salidas posibles (0, 1, 2, 3, 4, 5 dedos)
        )

    def forward(self, x):
        return self.red(x)


# --- CONFIGURACIÓN DEL ENTRENAMIENTO ---

# Cargar datos
dataset = DatasetManos('dataset_manos.csv')
# DataLoader agrupa los datos de 32 en 32 y los mezcla
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instanciar el modelo y enviarlo a la GPU
modelo = DetectorDeDedos().to(device)

# Función de pérdida (ideal para clasificación) y Optimizador
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# --- BUCLE DE ENTRENAMIENTO ---
epocas = 20  # Cuántas veces verá el dataset completo

print("Iniciando entrenamiento...")
for epoca in range(epocas):
    perdida_total = 0.0

    for batch_x, batch_y in dataloader:
        # ¡IMPORTANTE! Enviar los datos del batch a la GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 1. Limpiar gradientes anteriores
        optimizador.zero_grad()

        # 2. Predicción (Forward pass)
        predicciones = modelo(batch_x)

        # 3. Calcular el error
        perdida = criterio(predicciones, batch_y)

        # 4. Aprender del error (Backward pass)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item()

    # Imprimir el progreso
    perdida_promedio = perdida_total / len(dataloader)
    print(f"Época [{epoca + 1}/{epocas}], Pérdida: {perdida_promedio:.4f}")

print("¡Entrenamiento finalizado!")

# Guardar el modelo entrenado
torch.save(modelo.state_dict(), "modelo_dedos.pth")
print("Modelo guardado como 'modelo_dedos.pth'")