import rasterio
import numpy as np

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

# Ejemplo de uso:
ruta_archivo = 'path_to_your_elevation_data.tif'
gradiente_x, gradiente_y, pendiente = analizar_topografia(ruta_archivo)

# Imprimir los resultados
print("Gradiente X:\n", gradiente_x)
print("Gradiente Y:\n", gradiente_y)
print("Pendiente:\n", pendiente)
