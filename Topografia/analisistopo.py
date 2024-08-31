import numpy as np
import matplotlib.pyplot as plt
import rasterio
from mpl_toolkits.mplot3d import Axes3D

def plot_terrain_gradient(geotiff_path):
    """
    Esta función calcula el gradiente de un terreno a partir de una imagen GeoTIFF
    y lo despliega en una gráfica 3D de superficie y una gráfica 2D de curvas de nivel.

    :param geotiff_path: Ruta al archivo GeoTIFF que contiene los datos de elevación.
    """

    # Abrir la imagen GeoTIFF usando Rasterio
    with rasterio.open(geotiff_path) as dataset:
        # Leer la primera banda de la imagen (asumiendo que es un DEM)
        elevation = dataset.read(1)
        # Obtener las coordenadas de los píxeles
        transform = dataset.transform
        x = np.arange(elevation.shape[1]) * transform[0] + transform[2]
        y = np.arange(elevation.shape[0]) * transform[4] + transform[5]
        X, Y = np.meshgrid(x, y)

    # Calcular el gradiente del terreno
    dz_dx, dz_dy = np.gradient(elevation)
    gradient_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)

    # Crear la gráfica 3D de la superficie
    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: Gráfico 3D de la superficie
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, elevation, cmap='terrain', edgecolor='none')
    ax1.set_title('Superficie del Terreno')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Elevación (Z)')

    # Subplot 2: Gráfico 2D de curvas de nivel con gradiente
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, elevation, levels=20, cmap='terrain')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_title('Curvas de Nivel con Gradiente')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Superponer el gradiente en la gráfica de curvas de nivel
    ax2.imshow(gradient_magnitude, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='upper', cmap='Reds', alpha=0.5)

    plt.show()

# Ejemplo de uso:
plot_terrain_gradient('output_NASADEM.tif')
