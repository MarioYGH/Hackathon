import rasterio
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


def analizar_topografia(ruta_archivo):
    """
    Realiza un análisis topográfico de una zona a partir de un archivo rasterizado de datos de elevación.
    
    Args:
        ruta_archivo (str): Ruta al archivo rasterizado de elevación (.tif).
    
    Returns:
        gradiente_x (np.ndarray): Gradiente en la dirección X.
        gradiente_y (np.ndarray): Gradiente en la dirección Y.
        pendiente (np.ndarray): Pendiente del terreno calculada a partir de los gradientes.
    """
    # Leer el archivo rasterizado
    with rasterio.open(ruta_archivo) as src:
        elevacion = src.read(1)  # Leer el primer banda

    # Calcular los gradientes
    gradiente_x, gradiente_y = np.gradient(elevacion)

    # Calcular la pendiente
    pendiente = np.sqrt(gradiente_x**2 + gradiente_y**2)

    return gradiente_x, gradiente_y, pendiente

def graficar_gradientes(gradiente_x, gradiente_y):
    """
    Grafica los gradientes en las direcciones X e Y de un análisis topográfico.

    Args:
        gradiente_x (np.ndarray): Gradiente en la dirección X.
        gradiente_y (np.ndarray): Gradiente en la dirección Y.
    """
    # Crear una figura con dos subgráficas
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Graficar el gradiente en X
    cax1 = axs[0].imshow(gradiente_x, cmap='viridis')
    axs[0].set_title('Gradiente en X')
    fig.colorbar(cax1, ax=axs[0])

    # Graficar el gradiente en Y
    cax2 = axs[1].imshow(gradiente_y, cmap='viridis')
    axs[1].set_title('Gradiente en Y')
    fig.colorbar(cax2, ax=axs[1])

    plt.tight_layout()
    plt.show()
    

def graficar_superficie_y_gradientes_3d(gradiente_x, gradiente_y, pendiente):
    """
    Grafica la superficie continua de pendiente y el campo de gradientes en 3D en gráficos separados.

    Args:
        gradiente_x (np.ndarray): Gradiente en la dirección X.
        gradiente_y (np.ndarray): Gradiente en la dirección Y.
        pendiente (np.ndarray): Pendiente del terreno.
    """
    # Crear una figura con dos subgráficas
    fig = plt.figure(figsize=(18, 12))
    
    # Subgráfica 1: Superficie continua y contornos
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(np.arange(gradiente_x.shape[1]), np.arange(gradiente_x.shape[0]))
    Z = pendiente
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    cont = ax1.contour3D(X, Y, Z, 50, cmap='viridis', alpha=0.5)
    ax1.set_title('Superficie de Pendiente y Gradientes en 3D')
    ax1.set_xlabel('Dirección X')
    ax1.set_ylabel('Dirección Y')
    ax1.set_zlabel('Pendiente')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Subgráfica 2: Campo de gradientes
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(X, Y, Z, gradiente_x, gradiente_y, np.zeros_like(gradiente_x), length=1, normalize=True, color='blue')
    ax2.set_title('Campo de Gradientes en 3D')
    ax2.set_xlabel('Dirección X')
    ax2.set_ylabel('Dirección Y')
    ax2.set_zlabel('Pendiente')
    
    # Mostrar la figura con ambas subgráficas
    plt.show()

        
# viz.NASADEM_hillshade.tif
# output_SRTMGL3.tif
# output_NASADEM.tif
ruta_archivo = 'output_NASADEM.tif'
gradiente_x, gradiente_y, pendiente = analizar_topografia(ruta_archivo)

graficar_gradientes(gradiente_x, gradiente_y)
graficar_superficie_y_gradientes_3d(gradiente_x, gradiente_y, pendiente)


# Imprimir los resultados
print("Gradiente X:\n", gradiente_x)
print("Gradiente Y:\n", gradiente_y)
print("Pendiente:\n", pendiente)

