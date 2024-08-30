import rasterio
import numpy as np
import matplotlib.pyplot as plt 

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
    

# Ejemplo de uso:
ruta_archivo = 'output_NASADEM.tif'
gradiente_x, gradiente_y, pendiente = analizar_topografia(ruta_archivo)

graficar_gradientes(gradiente_x, gradiente_y)

# Imprimir los resultados
print("Gradiente X:\n", gradiente_x)
print("Gradiente Y:\n", gradiente_y)
print("Pendiente:\n", pendiente)
