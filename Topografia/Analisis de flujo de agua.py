import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

# Paso 1: Cargar la imagen GeoTIFF
with rasterio.open('output_NASADEM.tif') as dem:
    elevation = dem.read(1)  # Leer la primera banda (elevación)
    transform = dem.transform  # Transformación geoespacial

# Paso 2: Calcular el gradiente (pendiente)
gy, gx = np.gradient(elevation)
slope = np.sqrt(gx**2 + gy**2)  # Magnitud del gradiente (pendiente)

# Normalizar la pendiente entre 0 y 1
slope_normalized = (slope - slope.min()) / (slope.max() - slope.min())

# Paso 3: Calcular la dirección del flujo (opcional) y la acumulación de flujo
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

# Paso 4: Identificar los n puntos con máximo flujo
n = 10  # Puedes ajustar el número de puntos que quieres mostrar
max_flow_indices = np.unravel_index(np.argsort(flow_accumulation.ravel())[-n:], flow_accumulation.shape)
max_flow_points = list(zip(max_flow_indices[0], max_flow_indices[1]))

# Normalizar las coordenadas de los puntos de máximo flujo
rows, cols = flow_accumulation.shape
normalized_max_flow_points = [(x / rows, y / cols) for x, y in max_flow_points]

# Paso 5: Visualización de Resultados

plt.figure(figsize=(16, 14))

# Mapa de nivel
plt.subplot(2, 3, 1)
plt.title('Mapa de Nivel')
plt.imshow(elevation, cmap='terrain')
plt.colorbar(label='Elevación (m)')

# Mapa de pendiente
plt.subplot(2, 3, 2)
plt.title('Mapa de Pendiente')
plt.imshow(slope_normalized, cmap='viridis')
plt.colorbar(label='Pendiente Normalizada')

# Mapa de zonas de acumulación
plt.subplot(2, 3, 3)
plt.title('Zonas de Acumulación')
plt.imshow(flow_accumulation, cmap='Blues')
plt.colorbar(label='Acumulación de Flujo')

# Mapa de calor del flujo normalizado superpuesto con la pendiente normalizada
plt.subplot(2, 3, 4)
plt.title('Superposición de Pendiente Normalizada y Flujo Normalizado')
plt.imshow(slope_normalized, cmap='viridis', alpha=0.7)  # Mapa de pendiente normalizada con transparencia
plt.imshow(flow_normalized, cmap='hot', alpha=0.5)  # Mapa de flujo normalizado con transparencia
plt.colorbar(label='Pendiente Normalizada y Flujo Normalizado')

# Mapa de puntos de máximo flujo
plt.subplot(2, 3, 5)
plt.title(f'Top {n} Puntos de Máximo Flujo (Normalizado)')
plt.imshow(flow_normalized, cmap='hot')
plt.colorbar(label='Flujo Normalizado')
for point in normalized_max_flow_points:
    plt.plot(point[1] * cols, point[0] * rows, 'ro')  # Marcar puntos de máximo flujo en rojo, con coordenadas normalizadas

plt.tight_layout()
plt.show()
