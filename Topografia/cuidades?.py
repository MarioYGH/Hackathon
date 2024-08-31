import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

# Función para clasificar la estructura de recolección de agua basada en la magnitud del gradiente normalizado
def classify_structure(magnitude_normalized):
    if magnitude_normalized <= 0.05:
        return "Estanques"
    elif 0.05 < magnitude_normalized <= 0.1:
        return "Medias lunas"
    elif 0.1 < magnitude_normalized <= 0.15:
        return "Zanjas de infiltración"
    else:
        return "Terraza"

# Paso 1: Cargar la imagen GeoTIFF
with rasterio.open('/content/output_NASADEM_2.tif') as dem:
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
n = 100  # Puedes ajustar el número de puntos que quieres mostrar
max_flow_indices = np.unravel_index(np.argsort(flow_accumulation.ravel())[-n:], flow_accumulation.shape)
max_flow_points = list(zip(max_flow_indices[0], max_flow_indices[1]))

# Normalizar las coordenadas de los puntos de máximo flujo
rows, cols = flow_accumulation.shape
normalized_max_flow_points = [(x / rows, y / cols) for x, y in max_flow_points]

# Clasificar las estructuras de recolección de agua para los puntos de máximo flujo
classified_points = []
for y, x in max_flow_points:
    coord_x = x * transform[0] + transform[2]
    coord_y = y * transform[4] + transform[5]
    magnitude_normalized = slope_normalized[y, x]
    structure = classify_structure(magnitude_normalized)
    classified_points.append((coord_x, coord_y, magnitude_normalized, structure))


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
plt.imshow(normalized_max_flow_points, cmap='hot')
plt.colorbar(label='Flujo Normalizado')
for point in normalized_max_flow_points:
    plt.plot(point[1] * cols, point[0] * rows, 'ro')  # Marcar puntos de máximo flujo en rojo, con coordenadas normalizadas




# Inicializar una lista para almacenar las coordenadas y la estructura correspondiente
table_data = []


# Mostrar la gráfica
plt.tight_layout()
plt.show()

# Crear una tabla con las coordenadas y las estructuras
df = pd.DataFrame(table_data, columns=["Punto", "Coordenada X", "Coordenada Y", "Estructura"])

# Mostrar la tabla en la consola
print(df)

df.to_csv('tabla_coordenadas_estructuras.csv', index=False)

# Crear una nueva figura interactiva con Plotly
fig = go.Figure()

# Agregar la imagen de elevación como mapa de fondo
fig.add_trace(go.Heatmap(
    z=elevation,
    x=np.linspace(0, elevation.shape[1], elevation.shape[1]),
    y=np.linspace(0, elevation.shape[0], elevation.shape[0]),
    colorscale='spectral',
    reversescale=True,
    showscale=False
))


# Crear una tabla con las coordenadas y las estructuras
df = pd.DataFrame(table_data, columns=["Punto", "Coordenada X", "Coordenada Y", "Estructura"])
print(df)



# Imprimir las clasificaciones para los puntos aleatorios
for i, (coord_x, coord_y, magnitude_norm, structure) in enumerate(classified_points):
    print(f"Punto {i+1}: Coordenadas (X, Y) = ({coord_x}, {coord_y}), Magnitud del gradiente normalizada (0 a 1): {magnitude_norm}, Estructura recomendada: {structure}")
