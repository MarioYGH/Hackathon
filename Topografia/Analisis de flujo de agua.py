import rasterio
import numpy as np

# Cargar la imagen GeoTIFF
with rasterio.open('output_NASADEM.tif') as dem:
    elevation = dem.read(1)  # Leer la primera banda (elevación)
    transform = dem.transform  # Transformación geoespacial

# Calcular el gradiente en las direcciones x e y
gy, gx = np.gradient(elevation)
slope = np.sqrt(gx**2 + gy**2)  # Magnitud del gradiente (pendiente)

from scipy.ndimage import generic_filter

# Función para calcular la dirección del flujo
def calculate_flow_direction(gx, gy):
    return np.arctan2(gy, gx)

flow_direction = calculate_flow_direction(gx, gy)

# Inicializar la acumulación de flujo
flow_accumulation = np.ones_like(elevation)

# Función para acumular flujo
def accumulate_flow(window):
    center = window[4]
    flow_in = 0
    for i, direction in enumerate(window):
        if direction == center:
            flow_in += 1
    return flow_in

# Aplicar la función de acumulación sobre la matriz de dirección de flujo
flow_accumulation = generic_filter(flow_direction, accumulate_flow, size=3)

# Normalizar el flujo para visualizarlo mejor en un mapa de calor
flow_normalized = (flow_accumulation - flow_accumulation.min()) / (flow_accumulation.max() - flow_accumulation.min())

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 12))

# Mapa de nivel
plt.subplot(2, 2, 1)
plt.title('Mapa de Nivel')
plt.imshow(elevation, cmap='terrain')
plt.colorbar(label='Elevación (m)')

# Mapa de pendiente
plt.subplot(2, 2, 2)
plt.title('Mapa de Pendiente')
plt.imshow(slope, cmap='viridis')
plt.colorbar(label='Pendiente (°)')

# Mapa de zonas de acumulación
plt.subplot(2, 2, 3)
plt.title('Zonas de Acumulación')
plt.imshow(flow_accumulation, cmap='Blues')
plt.colorbar(label='Acumulación de Flujo')

# Mapa de calor del flujo normalizado
plt.subplot(2, 2, 4)
plt.title('Mapa de Calor de Flujo Normalizado')
plt.imshow(flow_normalized, cmap='hot')
plt.colorbar(label='Flujo Normalizado')

plt.tight_layout()
plt.show()

# Crear una figura
plt.figure(figsize=(10, 8))

# Mapa base de pendiente
plt.title('Superposición de Pendiente y Flujo Normalizado')
plt.imshow(slope, cmap='viridis', alpha=0.7)  # Mapa de pendiente con transparencia

# Superponer el mapa de flujo normalizado
plt.imshow(flow_normalized, cmap='hot', alpha=0.5)  # Mapa de flujo normalizado con transparencia

# Añadir una barra de color que combine ambas capas
cbar = plt.colorbar()
cbar.set_label('Pendiente y Flujo Normalizado')

plt.show()
