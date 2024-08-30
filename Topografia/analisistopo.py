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
    

def graficar_gradientes_3d(gradiente_x, gradiente_y, pendiente):
    # """
    # Grafica los gradientes en 3D utilizando flechas para mostrar la dirección y magnitud del gradiente.

    # Args:
    #     gradiente_x (np.ndarray): Gradiente en la dirección X.
    #     gradiente_y (np.ndarray): Gradiente en la dirección Y.
    #     pendiente (np.ndarray): Pendiente del terreno para la visualización en 3D.
    # """
    # # Crear una figura y un eje 3D
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # # Crear una malla de coordenadas
    # x = np.arange(gradiente_x.shape[1])
    # y = np.arange(gradiente_x.shape[0])
    # X, Y = np.meshgrid(x, y)
    
    # # Graficar la superficie del terreno
    # ax.plot_surface(X, Y, pendiente, cmap='terrain', alpha=0.8)

    # # Graficar los gradientes como flechas
    # ax.quiver(X, Y, pendiente, gradiente_x, gradiente_y, np.zeros_like(gradiente_x), length=1, color='blue')

    # # Configurar el título y las etiquetas
    # ax.set_title('Campo de Gradientes en 3D')
    # ax.set_xlabel('Dirección X')
    # ax.set_ylabel('Dirección Y')
    # ax.set_zlabel('Pendiente')

    # plt.show()
    
    # Crear una figura y un eje 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear una malla de coordenadas
    X, Y = np.meshgrid(np.arange(gradiente_x.shape[1]), np.arange(gradiente_x.shape[0]))
    Z = pendiente
    
    # Graficar el campo de gradientes como flechas en 3D
    ax.quiver(X, Y, Z, gradiente_x, gradiente_y, np.zeros_like(gradiente_x), length=1, normalize=True, color='blue')
    
    # Configurar el título y las etiquetas
    ax.set_title('Campo de Gradientes en 3D')
    ax.set_xlabel('Dirección X')
    ax.set_ylabel('Dirección Y')
    ax.set_zlabel('Pendiente')
    
    # Mostrar la gráfica
    plt.show()
    




        
# viz.NASADEM_hillshade.tif
# output_SRTMGL3.tif
# output_NASADEM.tif
ruta_archivo = 'output_NASADEM.tif'
gradiente_x, gradiente_y, pendiente = analizar_topografia(ruta_archivo)

graficar_gradientes(gradiente_x, gradiente_y)
graficar_gradientes_3d(gradiente_x, gradiente_y, pendiente)

# Imprimir los resultados
print("Gradiente X:\n", gradiente_x)
print("Gradiente Y:\n", gradiente_y)
print("Pendiente:\n", pendiente)

