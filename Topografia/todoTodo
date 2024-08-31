import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import matplotlib.patches as mpatches
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'

# Parámetros de la región
precipitation = 0.01  # Precipitación anual promedio en mm
cell_size = 30  # Tamaño de celda en metros (ajustar según tu DEM)
P_avg = 0.01  # Precipitación media en m/s (equivale a 36 mm/h)
T = 25  # Temperatura en grados Celsius
RH = 60  # Humedad relativa en %

# Convertir precipitación a metros
precipitation_m = precipitation / 1000  # mm a metros

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

# Definir parámetros físicos y funciones relevantes (conductividad hidráulica, etc.)
def K(theta):
    # Relación entre saturación y conductividad hidráulica (modelo simplificado)
    K_sat = 1e-6  # Conductividad hidráulica saturada en m/s (típico para suelos arcillosos)
    return K_sat * theta ** 2  # Ejemplo simple, conductividad cuadrática con saturación

def E(theta, T, RH):
    # Relación para la evaporación (simplificada)
    E_max = 5e-7  # Máxima tasa de evaporación en m/s (0.5 mm/h)
    return E_max * (1 - theta) * (T / 25) * (RH / 100)

def infiltration_area(scaling_factor, P_avg, Q_in):
    # Dimensiones proporcionales
    w = scaling_factor
    d = scaling_factor
    l = scaling_factor

    A = w * l
    V = w * l * d

    # Suponer una saturación inicial (theta inicial)
    theta = 0.3  # Saturación inicial del suelo, típica para suelos parcialmente secos
    grad_psi_z = -1  # Supuesto gradiente matricial simple

    # Calcular la tasa de infiltración considerando el flujo volumétrico que llega a la zanja
    infiltration = (K(theta) * grad_psi_z + (P_avg + Q_in) / A) - E(theta, T, RH)

    # Multiplicar por el área para obtener la infiltración total
    return infiltration * A

# Función objetivo para optimización (maximizar la infiltración)
def objective(scaling_factor, Q_in):
    return -infiltration_area(scaling_factor, P_avg, Q_in)  # Usamos el negativo para maximizar

# Función principal que optimiza las dimensiones de la zanja
def optimize_trench(Q_in):
    # Restricciones geométricas y físicas
    V_min = 1.0  # Volumen mínimo en m³
    A_max = 100.0  # Área máxima en m²

    cons = [{'type': 'ineq', 'fun': lambda s: s * (2 * s) * (3 * s) - V_min},  # Volumen mínimo
            {'type': 'ineq', 'fun': lambda s: A_max - s * (3 * s)}]  # Área máxima

    # Parámetros iniciales (suponer un punto de partida)
    initial_guess = [0.5]  # Escalamiento inicial en metros

    # Optimización
    result = minimize(objective, initial_guess, args=(Q_in,), constraints=cons, method='SLSQP')
    optimal_scaling_factor = result.x[0]

    # Dimensiones óptimas basadas en el factor de escalamiento
    optimal_w = optimal_scaling_factor
    optimal_d = 2 * optimal_scaling_factor
    optimal_l = 3 * optimal_scaling_factor

    # Calcular el volumen óptimo
    optimal_volume = optimal_w * optimal_d * optimal_l

    # Devolver los resultados
    return {
        'optimal_w': optimal_w,
        'optimal_d': optimal_d,
        'optimal_l': optimal_l,
        'optimal_volume': optimal_volume,
        'maximum_infiltration': -result.fun
    }

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

# Clasificar las estructuras de recolección de agua para los puntos de máximo flujo y calcular volúmenes
classified_points = []
for y, x in max_flow_points:
    coord_x = x * transform[0] + transform[2]
    coord_y = y * transform[4] + transform[5]
    magnitude_normalized = slope_normalized[y, x]
    structure = classify_structure(magnitude_normalized)
    
    # Calcular el área de captación en metros cuadrados solo para los puntos seleccionados
    area = flow_accumulation[y, x] * (cell_size ** 2)
    
    # Calcular el flujo volumétrico basado en la precipitación y el área de captación solo para los puntos seleccionados
    Q_in = area * precipitation_m * flow_normalized[y, x]
    
    # Optimizar el volumen de la zanja usando el flujo volumétrico calculado
    optimization_results = optimize_trench(Q_in)
    
    classified_points.append({
        'coord_x': coord_x,
        'coord_y': coord_y,
        'magnitude_normalized': magnitude_normalized,
        'structure': structure,
        'Q_in': Q_in,
        'optimal_w': optimization_results['optimal_w'],
        'optimal_d': optimization_results['optimal_d'],
        'optimal_l': optimization_results['optimal_l'],
        'optimal_volume': optimization_results['optimal_volume'],
        'maximum_infiltration': optimization_results['maximum_infiltration']
    })

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

# Mapa de puntos de máximo flujo con representación de estructuras y volúmenes
plt.subplot(2, 3, 6)
plt.title(f'Top {n} Puntos de Máximo Flujo con Volúmenes')
plt.imshow(elevation, cmap='terrain')
plt.colorbar(label='Elevación (m)')
for i, point in enumerate(classified_points):
    x_plot = (point['coord_x'] - transform[2]) / transform[0]
    y_plot = (point['coord_y'] - transform[5]) / transform[4]
    
    if point['structure'] == "Estanques":
        plt.plot(x_plot, y_plot, 'bo', markersize=10)  # Círculo azul para estanques
    elif point['structure'] == "Medias lunas":
        plt.plot(x_plot, y_plot, 'ko', markersize=12)  # Círculo negro más grande
        plt.plot([x_plot - 3, x_plot + 3], [y_plot, y_plot], 'k-', lw=3)  # Línea más larga para simular una media luna
    elif point['structure'] == "Zanjas de infiltración":
        plt.plot([x_plot - 2, x_plot + 2], [y_plot, y_plot], 'r-', lw=2)  # Línea roja para zanjas de infiltración
    else:
        plt.plot([x_plot - 2, x_plot + 2], [y_plot + 2, y_plot + 2], 'g-', lw=2)  # Línea verde para terrazas
        plt.plot([x_plot - 2, x_plot + 2], [y_plot - 2, y_plot - 2], 'g-', lw=2)  # Segunda línea para simular una terraza

    # Mostrar el volumen optimizado en la gráfica
    plt.text(x_plot, y_plot, f'{point["optimal_volume"]:.1f} m³', fontsize=8, color='white', ha='center')

# Crear los parches para la leyenda
estanque_patch = mpatches.Patch(color='blue', label='Estanques')
media_luna_patch = mpatches.Patch(color='black', label='Medias lunas')
zanja_patch = mpatches.Patch(color='red', label='Zanjas')
terraza_patch = mpatches.Patch(color='green', label='Terrazas')

# Añadir la leyenda a la gráfica
plt.legend(handles=[estanque_patch, media_luna_patch, zanja_patch, terraza_patch], loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# Crear una tabla con las coordenadas, estructuras y volúmenes
df = pd.DataFrame(classified_points)

# Guardar la tabla en un archivo CSV
df.to_csv('tabla_coordenadas_estructuras_volumen.csv', index=False)

# Mostrar la tabla en la consola
print(df)

# Imprimir las clasificaciones para los puntos seleccionados
for i, point in enumerate(classified_points):
    print(f"Punto {i+1}: Coordenadas (X, Y) = ({point['coord_x']}, {point['coord_y']}), "
          f"Estructura recomendada: {point['structure']}, "
          f"Q_in: {point['Q_in']:.6f} m³/s, "
          f"Volumen optimizado: {point['optimal_volume']:.1f} m³, "
          f"Máxima infiltración: {point['maximum_infiltration']:.4f} m³/s")

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
for i, point in enumerate(classified_points):
    x_plot = (point['coord_x'] - transform[2]) / transform[0]
    y_plot = (point['coord_y'] - transform[5]) / transform[4]
    
    if point['structure'] == "Estanques":
        fig.add_trace(go.Scatter(
            x=[x_plot], y=[y_plot], 
            mode='markers+text',
            text=[str(i+1)],
            textposition='top center',
            marker=dict(color='blue', size=10),
            name=f'Estanque {i+1}'
        ))
    elif point['structure'] == "Medias lunas":
        fig.add_trace(go.Scatter(
            x=[x_plot], y=[y_plot], 
            mode='markers+text',
            text=[str(i+1)],
            textposition='top center',
            marker=dict(color='black', size=10),
            name=f'Media luna {i+1}'
        ))
    elif point['structure'] == "Zanjas de infiltración":
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
