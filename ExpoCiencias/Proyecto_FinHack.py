import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
#from mpl_toolkits.mplot3d import Axes3D
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

# Mapa de puntos de máximo flujo con representación de estructuras
plt.subplot(2, 3, 6)
plt.title(f'Top {n} Puntos de Máximo Flujo (Normalizado)')
plt.imshow(elevation, cmap='terrain')
plt.colorbar(label='Elevación (m)')
for i, (coord_x, coord_y, magnitude_norm, structure) in enumerate(classified_points):
    x_plot = (coord_x - transform[2]) / transform[0]
    y_plot = (coord_y - transform[5]) / transform[4]
    
    if structure == "Estanques":
        plt.plot(x_plot, y_plot, 'bo', markersize=10)  # Círculo azul para estanques
    elif structure == "Medias lunas":
        plt.plot(x_plot, y_plot, 'ko', markersize=12)  # Círculo negro más grande
        plt.plot([x_plot - 3, x_plot + 3], [y_plot, y_plot], 'k-', lw=3)  # Línea más larga para simular una media luna
    elif structure == "Zanjas de infiltración":
        plt.plot([x_plot - 2, x_plot + 2], [y_plot, y_plot], 'r-', lw=2)  # Línea roja para zanjas de infiltración
    else:
        plt.plot([x_plot - 2, x_plot + 2], [y_plot + 2, y_plot + 2], 'g-', lw=2)  # Línea verde para terrazas
        plt.plot([x_plot - 2, x_plot + 2], [y_plot - 2, y_plot - 2], 'g-', lw=2)  # Segunda línea para simular una terraza

# Crear los parches para la leyenda
estanque_patch = mpatches.Patch(color='blue', label='Estanques')
media_luna_patch = mpatches.Patch(color='black', label='MLunas')
zanja_patch = mpatches.Patch(color='red', label='Zanjas')
terraza_patch = mpatches.Patch(color='green', label='Terrazas')

# Añadir la leyenda a la gráfica
plt.legend(handles=[estanque_patch, media_luna_patch, zanja_patch, terraza_patch], loc='upper right', fontsize=10)


plt.tight_layout()
plt.show()

# Crear una nueva figura más grande
plt.figure(figsize=(20, 16))

# Mapa de puntos de máximo flujo con representación de estructuras (gráfico grande)
plt.title(f'Top {n} Puntos de Máximo Flujo (Normalizado)')
plt.imshow(elevation, cmap='terrain')
plt.colorbar(label='Elevación (m)')

# Inicializar una lista para almacenar las coordenadas y la estructura correspondiente
table_data = []

# Iterar sobre los puntos clasificados para graficar cada estructura
for i, (coord_x, coord_y, magnitude_norm, structure) in enumerate(classified_points):
    x_plot = (coord_x - transform[2]) / transform[0]
    y_plot = (coord_y - transform[5]) / transform[4]
    
    # Graficar cada punto con su símbolo correspondiente
    if structure == "Estanques":
        plt.plot(x_plot, y_plot, 'bo', markersize=10)  # Círculo azul para estanques
    elif structure == "Medias lunas":
        plt.plot(x_plot, y_plot, 'ko', markersize=12)  # Círculo negro más grande
        plt.plot([x_plot - 3, x_plot + 3], [y_plot, y_plot], 'k-', lw=3)  # Línea más larga para simular una media luna
    elif structure == "Zanjas de infiltración":
        plt.plot([x_plot - 2, x_plot + 2], [y_plot, y_plot], 'r-', lw=2)  # Línea roja para zanjas de infiltración
    else:
        plt.plot([x_plot - 2, x_plot + 2], [y_plot + 2, y_plot + 2], 'g-', lw=2)  # Línea verde para terrazas
        plt.plot([x_plot - 2, x_plot + 2], [y_plot - 2, y_plot - 2], 'g-', lw=2)  # Segunda línea para simular una terraza

    # Numerar el punto en la gráfica
    plt.text(x_plot, y_plot, str(i+1), fontsize=9, color='white', ha='center', va='center')

    # Agregar las coordenadas y la estructura a la tabla
    table_data.append([i+1, coord_x, coord_y, structure])

# Crear los parches para la leyenda
estanque_patch = mpatches.Patch(color='blue', label='Estanques')
media_luna_patch = mpatches.Patch(color='black', label='Medias lunas')
zanja_patch = mpatches.Patch(color='red', label='Zanjas de infiltración')
terraza_patch = mpatches.Patch(color='green', label='Terrazas')

# Añadir la leyenda a la gráfica
plt.legend(handles=[estanque_patch, media_luna_patch, zanja_patch, terraza_patch], loc='upper right', fontsize=12)

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

# Iterar sobre los puntos clasificados para graficar cada estructura
for i, (coord_x, coord_y, magnitude_norm, structure) in enumerate(classified_points):
    x_plot = (coord_x - transform[2]) / transform[0]
    y_plot = (coord_y - transform[5]) / transform[4]
    
    if structure == "Estanques":
        fig.add_trace(go.Scatter(
            x=[x_plot], y=[y_plot], 
            mode='markers+text',
            text=[str(i+1)],
            textposition='top center',
            marker=dict(color='blue', size=10),
            name=f'Estanque {i+1}'
        ))
    elif structure == "Medias lunas":
        fig.add_trace(go.Scatter(
            x=[x_plot], y=[y_plot], 
            mode='markers+text',
            text=[str(i+1)],
            textposition='top center',
            marker=dict(color='black', size=10),
            name=f'Media luna {i+1}'
        ))
    elif structure == "Zanjas de infiltración":
        fig.add_trace(go.Scatter(
            x=[x_plot], y=[y_plot],
            mode='markers+text',
            text=[str(i+1)],
            textposition='top center',
            marker=dict(color='red', size=10),
            name=f'Zanja {i+1}'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[x_plot, x_plot],
            y=[y_plot + 2, y_plot - 2],
            mode='markers+text',
            text=[str(i+1)],
            textposition='top center',
            marker=dict(color='green', size=10),
            name=f'Terraza {i+1}'
        ))

# Ajustar el layout para mejorar la visualización
fig.update_layout(
    title="Top Puntos de Máximo Flujo (Normalizado) con Estructuras",
    xaxis_title="Coordenada X",
    yaxis_title="Coordenada Y",
    showlegend=True,
    height=1000,  # Aumentar el tamaño en altura
    width=1200,   # Aumentar el tamaño en anchura
    margin=dict(l=50, r=50, t=50, b=50)  # Ajustar márgenes para asegurarse de que todo esté visible
)

# Mostrar la gráfica interactiva
fig.show()

# Crear una tabla con las coordenadas y las estructuras
df = pd.DataFrame(table_data, columns=["Punto", "Coordenada X", "Coordenada Y", "Estructura"])
print(df)

# Imprimir las clasificaciones para los puntos aleatorios
for i, (coord_x, coord_y, magnitude_norm, structure) in enumerate(classified_points):
    print(f"Punto {i+1}: Coordenadas (X, Y) = ({coord_x}, {coord_y}), Magnitud del gradiente normalizada (0 a 1): {magnitude_norm}, Estructura recomendada: {structure}")
