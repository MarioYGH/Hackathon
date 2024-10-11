import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
import pandas as pd
from scipy.ndimage import generic_filter

# Configurar la apariencia de CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Función para clasificar la estructura basada en la magnitud del gradiente normalizado
def classify_structure(magnitude_normalized):
    if magnitude_normalized <= 0.05:
        return "Estanques"
    elif 0.05 < magnitude_normalized <= 0.1:
        return "Medias lunas"
    elif 0.1 < magnitude_normalized <= 0.15:
        return "Zanjas de infiltración"
    else:
        return "Terraza"

# Función para cargar el archivo GeoTIFF
def cargar_imagen():
    filepath = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif")])
    if filepath:
        with rasterio.open(filepath) as dem:
            global elevation
            elevation = dem.read(1)  # Leer la primera banda (elevación)
            global transform
            transform = dem.transform  # Transformación geoespacial
        label_cargar.configure(text=f"Imagen cargada: {filepath}")
        btn_mostrar_graficas.configure(state="normal")  # Habilitar botón

# Función para mostrar las gráficas en la segunda pestaña
def mostrar_graficas():
    global classified_points, n, slope_normalized
    # Calcular la pendiente
    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gx ** 2 + gy ** 2)  # Magnitud del gradiente (pendiente)
    slope_normalized = (slope - slope.min()) / (slope.max() - slope.min())  # Normalización

    # Calcular dirección de flujo
    flow_direction = np.arctan2(gy, gx)

    # Inicializar acumulación de flujo
    flow_accumulation = np.ones_like(elevation)

    # Función para acumular flujo
    def accumulate_flow(window):
        center = window[4]
        flow_in = 0
        for i, direction in enumerate(window):
            if direction == center:
                flow_in += 1
        return flow_in

    flow_accumulation = generic_filter(flow_direction, accumulate_flow, size=3)

    # Normalizar el flujo para visualizarlo mejor en un mapa de calor
    flow_normalized = (flow_accumulation - flow_accumulation.min()) / (flow_accumulation.max() - flow_accumulation.min())

    # Identificar los n puntos con máximo flujo
    n = 100
    max_flow_indices = np.unravel_index(np.argsort(flow_accumulation.ravel())[-n:], flow_accumulation.shape)
    max_flow_points = list(zip(max_flow_indices[0], max_flow_indices[1]))

    # Clasificar las estructuras de recolección de agua
    classified_points = []
    for y, x in max_flow_points:
        coord_x = x * transform[0] + transform[2]
        coord_y = y * transform[4] + transform[5]
        magnitude_normalized = slope_normalized[y, x]
        structure = classify_structure(magnitude_normalized)
        classified_points.append((coord_x, coord_y, magnitude_normalized, structure))

    # Crear el mapa interactivo con Plotly
    fig_interactive = go.Figure()

    # Añadir la capa de elevación
    fig_interactive.add_trace(go.Heatmap(
        z=elevation,
        colorscale='spectral',  # Usar 'spectral' para colores similares al ejemplo que diste
        reversescale=True,
        showscale=True
    ))

    # Añadir los puntos clasificados al mapa interactivo
    for i, (coord_x, coord_y, magnitude_norm, structure) in enumerate(classified_points):
        x_plot = (coord_x - transform[2]) / transform[0]
        y_plot = (coord_y - transform[5]) / transform[4]
        
        if structure == "Estanques":
            fig_interactive.add_trace(go.Scatter(
                x=[x_plot], y=[y_plot], 
                mode='markers+text',
                text=[str(i+1)],
                textposition='top center',
                marker=dict(color='blue', size=10),
                name=f'Estanque {i+1}'
            ))
        elif structure == "Medias lunas":
            fig_interactive.add_trace(go.Scatter(
                x=[x_plot], y=[y_plot], 
                mode='markers+text',
                text=[str(i+1)],
                textposition='top center',
                marker=dict(color='black', size=10),
                name=f'Media luna {i+1}'
            ))
        elif structure == "Zanjas de infiltración":
            fig_interactive.add_trace(go.Scatter(
                x=[x_plot], y=[y_plot],
                mode='markers+text',
                text=[str(i+1)],
                textposition='top center',
                marker=dict(color='red', size=10),
                name=f'Zanja {i+1}'
            ))
        else:
            fig_interactive.add_trace(go.Scatter(
                x=[x_plot, x_plot],
                y=[y_plot + 2, y_plot - 2],
                mode='markers+text',
                text=[str(i+1)],
                textposition='top center',
                marker=dict(color='green', size=10),
                name=f'Terraza {i+1}'
            ))

    # Ajustar el layout para mejorar la visualización
    fig_interactive.update_layout(
        title="Top Puntos de Máximo Flujo (Normalizado) con Estructuras",
        height=800,
        width=1200,
        showlegend=True
    )

    # Mostrar el mapa interactivo
    fig_interactive.show()

# Función para guardar los datos clasificados en un archivo CSV
def guardar_datos():
    if classified_points:
        # Crear DataFrame con los datos clasificados
        df = pd.DataFrame(classified_points, columns=["Coordenada X", "Coordenada Y", "Magnitud Normalizada", "Estructura"])
        # Pedir al usuario donde guardar el archivo CSV
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filepath:
            df.to_csv(filepath, index=False)
            label_guardar.configure(text=f"Datos guardados en: {filepath}")

# Crear ventana principal
root = ctk.CTk()
root.title("Análisis de Imágenes GeoTIFF")

# Crear pestañas
tabview = ctk.CTkTabview(root)
tabview.pack(fill="both", expand=True)

# Pestaña 1: Cargar Imagen
pestana_cargar = tabview.add("Cargar Imagen")
label_cargar = ctk.CTkLabel(pestana_cargar, text="Cargar imagen GeoTIFF para análisis", font=("Arial", 16))
label_cargar.pack(pady=20)

btn_cargar = ctk.CTkButton(pestana_cargar, text="Cargar Imagen", command=cargar_imagen)
btn_cargar.pack(pady=20)

# Pestaña 2: Mostrar Gráficas
pestana_graficas = tabview.add("Mostrar Gráficas")
btn_mostrar_graficas = ctk.CTkButton(pestana_graficas, text="Mostrar Gráficas", command=mostrar_graficas, state="disabled")
btn_mostrar_graficas.pack(pady=20)

# Botón para guardar datos en CSV
label_guardar = ctk.CTkLabel(pestana_graficas, text="", font=("Arial", 12))
label_guardar.pack(pady=5)

btn_guardar = ctk.CTkButton(pestana_graficas, text="Guardar Datos en CSV", command=guardar_datos)
btn_guardar.pack(pady=10)

# Ejecutar la interfaz
root.geometry("800x600")
root.mainloop()
