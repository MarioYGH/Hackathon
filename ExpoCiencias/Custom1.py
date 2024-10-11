import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
import pandas as pd
from PIL import ImageTk, Image

# Configurar la apariencia de CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Función para cargar el archivo GeoTIFF
def cargar_imagen():
    filepath = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif")])
    if filepath:
        with rasterio.open(filepath) as dem:
            global elevation
            elevation = dem.read(1)  # Leer la primera banda (elevación)
            global transform
            transform = dem.transform  # Transformación geoespacial
        # Reemplazar config() por configure()
        label_cargar.configure(text=f"Imagen cargada: {filepath}")
        btn_mostrar_graficas.configure(state="normal")  # Habilitar botón


# Función para mostrar las gráficas en la segunda pestaña
def mostrar_graficas():
    # Calcular la pendiente
    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gx ** 2 + gy ** 2)  # Magnitud del gradiente (pendiente)
    slope_normalized = (slope - slope.min()) / (slope.max() - slope.min())  # Normalización

    # Crear la figura de matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Mapa de elevación
    axs[0].imshow(elevation, cmap='terrain')
    axs[0].set_title('Mapa de Elevación')
    axs[0].axis('off')

    # Mapa de pendiente normalizada
    axs[1].imshow(slope_normalized, cmap='viridis')
    axs[1].set_title('Mapa de Pendiente Normalizada')
    axs[1].axis('off')

    # Insertar el gráfico en la ventana
    canvas = FigureCanvasTkAgg(fig, master=pestana_graficas)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Crear mapa interactivo con Plotly
    fig_interactive = go.Figure()

    fig_interactive.add_trace(go.Heatmap(
        z=elevation,
        colorscale='viridis',  # Usar 'viridis', 'earth', 'turbo', o cualquier otra escala válida de Plotly
        reversescale=True,
        showscale=True
    ))

    fig_interactive.update_layout(
        title="Mapa Interactivo de Elevación",
        height=600,
        width=800
    )
    fig_interactive.show()


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

# Ejecutar la interfaz
root.geometry("800x600")
root.mainloop()
