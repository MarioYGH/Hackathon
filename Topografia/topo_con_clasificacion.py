import numpy as np
import matplotlib.pyplot as plt
import rasterio
from mpl_toolkits.mplot3d import Axes3D

def classify_structure(magnitude_normalized):
    """
    Clasifica la estructura de recolección de agua basada en la magnitud del gradiente normalizado.

    :param magnitude_normalized: Magnitud del gradiente normalizado (de 0 a 1)
    :return: El tipo de estructura de recolección de agua recomendada.
    """
    if magnitude_normalized <= 0.2:
        return "Estanques"
    elif 0.2 < magnitude_normalized <= 0.5:
        return "Medias lunas"
        

    elif 0.5 < magnitude_normalized <= 0.7:
        return "Zanjas de infiltración"
        
    else:
        return "Terraza"

def plot_terrain_gradient_magnitude_normalized(geotiff_path, num_points=5):
    """
    Esta función calcula la magnitud del gradiente de un terreno a partir de una imagen GeoTIFF,
    normaliza la magnitud del gradiente en un rango de 0 a 1 para un número específico de puntos aleatorios,
    y clasifica las estructuras de recolección de agua para esos puntos.

    :param geotiff_path: Ruta al archivo GeoTIFF que contiene los datos de elevación.
    :param num_points: Número de puntos aleatorios en los que se calculará el gradiente.
    :return: Una lista de tuplas con las coordenadas (X, Y), la magnitud del gradiente normalizada, y la clasificación.
    """

    #Abrir la imagen GeoTIFF usando Rasterio
    with rasterio.open(geotiff_path) as dataset:
        #Leer la primera banda de la imagen (asumiendo que es un DEM)
        elevation = dataset.read(1)
        #Obtener las coordenadas de los píxeles
        transform = dataset.transform
        x = np.arange(elevation.shape[1]) * transform[0] + transform[2]
        y = np.arange(elevation.shape[0]) * transform[4] + transform[5]
        X, Y = np.meshgrid(x, y)

    #Calcular el gradiente del terreno
    dz_dx, dz_dy = np.gradient(elevation)
    gradient_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)

    #Encontrar el valor mínimo y máximo de la magnitud del gradiente
    magnitude_min, magnitude_max = gradient_magnitude.min(), gradient_magnitude.max()

    #Normalizar la magnitud del gradiente en un rango de 0 a 1
    gradient_magnitude_normalized = (gradient_magnitude - magnitude_min) / (magnitude_max - magnitude_min)

    #Generar puntos aleatorios
    points = []
    for _ in range(num_points):
        #Seleccionar índices aleatorios dentro del rango de la matriz
        idx_x = np.random.randint(0, elevation.shape[1])
        idx_y = np.random.randint(0, elevation.shape[0])

        #Obtener las coordenadas y la magnitud del gradiente normalizada en esos puntos
        coord_x = x[idx_x]
        coord_y = y[idx_y]
        magnitude_normalized = gradient_magnitude_normalized[idx_y, idx_x]

        #Clasificar la estructura de recolección de agua
        structure = classify_structure(magnitude_normalized)

        points.append((coord_x, coord_y, magnitude_normalized, structure))

    #Crear la gráfica 3D de la superficie
    fig = plt.figure(figsize=(14, 6))

    #Subplot 1: Gráfico 3D de la superficie
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, elevation, cmap='terrain', edgecolor='none')
    ax1.set_title('Superficie del Terreno')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Elevación (Z)')

    #Subplot 2: Gráfico 2D de curvas de nivel con gradiente
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, elevation, levels=20, cmap='terrain')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_title('Curvas de Nivel con Magnitud del Gradiente Normalizada de 0 a 1')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    plt.show()

    #Devolver los puntos y las clasificaciones
    return points

#Ejemplo de uso:
random_points_classified = plot_terrain_gradient_magnitude_normalized('/content/output_NASADEM.tif', num_points=5)

#Imprimir las clasificaciones para los puntos aleatorios
for i, (coord_x, coord_y, magnitude_norm, structure) in enumerate(random_points_classified):
    print(f"Punto {i+1}: Coordenadas (X, Y) = ({coord_x}, {coord_y}), Magnitud del gradiente normalizada (0 a 1): {magnitude_norm}, Estructura recomendada: {structure}")
