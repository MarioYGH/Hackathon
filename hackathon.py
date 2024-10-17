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
        label_imagen_cargada.configure(text=f"Imagen cargada: {filepath}")
        # Habilitar los botones en la pestaña "Cargar Imagen"
        btn_mostrar_graficas.configure(state="normal")
        btn_guardar.configure(state="normal")

# Función para mostrar las gráficas en la pestaña "Cargar Imagen"
def mostrar_graficas():
    global classified_points, n, slope_normalized
    try:
        n = int(entry_zanjas.get())  # Leer el número de zanjas especificado por el usuario
    except ValueError:
        label_guardar.configure(text="Por favor, ingresa un número válido de zanjas.")
        return

    if n <= 0:
        label_guardar.configure(text="El número de zanjas debe ser positivo.")
        return

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
        for direction in window:
            if direction == center:
                flow_in += 1
        return flow_in

    flow_accumulation = generic_filter(flow_direction, accumulate_flow, size=3)

    # Normalizar el flujo para visualizarlo mejor en un mapa de calor
    flow_normalized = (flow_accumulation - flow_accumulation.min()) / (
            flow_accumulation.max() - flow_accumulation.min())

    # Identificar los n puntos con máximo flujo
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

    # Limpiar cualquier gráfica anterior en la pestaña "Cargar Imagen"
    for widget in frame_graficas.winfo_children():
        widget.destroy()

    # Crear las gráficas estáticas con Matplotlib
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw={'hspace': 0.5})
    # Mapa de nivel
    axs[0, 0].set_title('Mapa de Nivel')
    im = axs[0, 0].imshow(elevation, cmap='terrain')
    fig.colorbar(im, ax=axs[0, 0], label='Elevación (m)')
    # Mapa de pendiente
    axs[0, 1].set_title('Mapa de Pendiente')
    im = axs[0, 1].imshow(slope_normalized, cmap='viridis')
    fig.colorbar(im, ax=axs[0, 1], label='Pendiente Normalizada')
    # Mapa de zonas de acumulación
    axs[0, 2].set_title('Zonas de Acumulación')
    im = axs[0, 2].imshow(flow_accumulation, cmap='Blues')
    fig.colorbar(im, ax=axs[0, 2], label='Acumulación de Flujo')
    # Mapa de superposición de pendiente y flujo
    axs[1, 0].set_title('Superposición Pendiente/Flujo')
    axs[1, 0].imshow(slope_normalized, cmap='viridis', alpha=0.7)
    im = axs[1, 0].imshow(flow_normalized, cmap='hot', alpha=0.5)
    fig.colorbar(im, ax=axs[1, 0], label='Pendiente Normalizada y Flujo')
    # Mapa de puntos de máximo flujo
    axs[1, 1].set_title(f'Top {n} Puntos de Máximo Flujo')
    axs[1, 1].imshow(elevation, cmap='terrain')
    for i, (coord_x, coord_y, magnitude_norm, structure) in enumerate(classified_points):
        x_plot = (coord_x - transform[2]) / transform[0]
        y_plot = (coord_y - transform[5]) / transform[4]
        axs[1, 1].plot(x_plot, y_plot, 'ro')
    # Eliminar el sexto subgráfico vacío
    fig.delaxes(axs[1, 2])

    # Ajustar el espacio entre subplots para evitar que los títulos se sobrepongan
    fig.tight_layout()

    # Insertar las gráficas estáticas en la pestaña "Cargar Imagen"
    canvas = FigureCanvasTkAgg(fig, master=frame_graficas)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

    # Crear el mapa interactivo con Plotly y mostrarlo en una nueva ventana
    fig_interactive = go.Figure()
    # Añadir la capa de elevación
    fig_interactive.add_trace(go.Heatmap(
        z=elevation,
        colorscale='spectral',
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
                text=[str(i + 1)],
                textposition='top center',
                marker=dict(color='blue', size=10),
                name=f'Estanque {i + 1}'
            ))
        elif structure == "Medias lunas":
            fig_interactive.add_trace(go.Scatter(
                x=[x_plot], y=[y_plot],
                mode='markers+text',
                text=[str(i + 1)],
                textposition='top center',
                marker=dict(color='black', size=10),
                name=f'Media luna {i + 1}'
            ))
        elif structure == "Zanjas de infiltración":
            fig_interactive.add_trace(go.Scatter(
                x=[x_plot], y=[y_plot],
                mode='markers+text',
                text=[str(i + 1)],
                textposition='top center',
                marker=dict(color='red', size=10),
                name=f'Zanja {i + 1}'
            ))
        else:
            fig_interactive.add_trace(go.Scatter(
                x=[x_plot], y=[y_plot],
                mode='markers+text',
                text=[str(i + 1)],
                textposition='top center',
                marker=dict(color='green', size=10),
                name=f'Terraza {i + 1}'
            ))
    # Ajustar el layout para mejorar la visualización
    fig_interactive.update_layout(
        title="Top Puntos de Máximo Flujo (Normalizado) con Estructuras",
        height=800,
        width=1200,
        showlegend=True
    )
    # Mostrar el mapa interactivo en una nueva ventana del navegador
    fig_interactive.show()

# Función para guardar los datos clasificados en un archivo CSV
def guardar_datos():
    if classified_points:
        # Crear DataFrame con los datos clasificados
        df = pd.DataFrame(classified_points,
                          columns=["Coordenada X", "Coordenada Y", "Magnitud Normalizada", "Estructura"])
        # Pedir al usuario dónde guardar el archivo CSV
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filepath:
            df.to_csv(filepath, index=False)
            label_guardar.configure(text=f"Datos guardados en: {filepath}")
    else:
        label_guardar.configure(text="No hay datos para guardar. Por favor, genera las gráficas primero.")

# Crear ventana principal
root = ctk.CTk()
root.title("Análisis de Imágenes GeoTIFF")
root.geometry("1000x700")

# Crear pestañas
tabview = ctk.CTkTabview(root)
tabview.pack(fill="both", expand=True)

# ---------------- Pestaña 1: Cargar Imagen ----------------
pestana_cargar = tabview.add("Cargar Imagen")

# Añadir título principal
titulo_principal = ctk.CTkLabel(pestana_cargar, text="Análisis de Imágenes GeoTIFF", font=("Arial", 24, "bold"))
titulo_principal.pack(pady=20)

# Crear un frame para los controles
frame_controles = ctk.CTkFrame(pestana_cargar)
frame_controles.pack(pady=10, padx=20, fill="x")

# Botón para cargar imagen y etiqueta de imagen cargada
btn_cargar = ctk.CTkButton(frame_controles, text="Cargar Imagen", command=cargar_imagen)
btn_cargar.grid(row=0, column=0, pady=10, padx=10)

label_imagen_cargada = ctk.CTkLabel(frame_controles, text="No se ha cargado ninguna imagen", font=("Arial", 14))
label_imagen_cargada.grid(row=0, column=1, pady=10, padx=10, sticky="w")

# Entrada para el número de zanjas
label_zanjas = ctk.CTkLabel(frame_controles, text="Número de zanjas:", font=("Arial", 14))
label_zanjas.grid(row=1, column=0, pady=10, padx=10, sticky="e")
entry_zanjas = ctk.CTkEntry(frame_controles, width=100, font=("Arial", 14))
entry_zanjas.grid(row=1, column=1, pady=10, padx=10, sticky="w")

# Frame para los botones de acción
frame_botones = ctk.CTkFrame(pestana_cargar)
frame_botones.pack(pady=10, padx=20, fill="x")

btn_mostrar_graficas = ctk.CTkButton(frame_botones, text="Mostrar Gráficas", command=mostrar_graficas,
                                     state="disabled", font=("Arial", 14))
btn_mostrar_graficas.grid(row=0, column=0, pady=10, padx=10)

btn_guardar = ctk.CTkButton(frame_botones, text="Guardar Datos en CSV", command=guardar_datos,
                            state="disabled", font=("Arial", 14))
btn_guardar.grid(row=0, column=1, pady=10, padx=10)

# Etiqueta para mensajes
label_guardar = ctk.CTkLabel(pestana_cargar, text="", font=("Arial", 14))
label_guardar.pack(pady=10)

# Frame para las gráficas
frame_graficas = ctk.CTkFrame(pestana_cargar)
frame_graficas.pack(pady=10, padx=20, fill="both", expand=True)

# ---------------- Pestaña 2: Información ----------------
pestana_informacion = tabview.add("Información")

# Crear un frame con scroll
scrollable_frame = ctk.CTkScrollableFrame(pestana_informacion, width=800)
scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Título principal en la pestaña Información
label_titulo = ctk.CTkLabel(scrollable_frame, text="Análisis Topográfico para Diseño de Estructuras Auxiliares",
                            font=("Arial", 24, "bold"))
label_titulo.pack(pady=(10, 20))

# Sección: Resumen del Proyecto
frame_resumen = ctk.CTkFrame(scrollable_frame)
frame_resumen.pack(fill="x", padx=10, pady=10)

label_seccion_resumen = ctk.CTkLabel(frame_resumen, text="Resumen del Proyecto", font=("Arial", 20, "bold"))
label_seccion_resumen.pack(pady=10)

resumen_proyecto = """
El proyecto tiene como objetivo restaurar los mantos freáticos mediante la recolección de agua en zanjas de infiltración y otras estructuras auxiliares. A través del análisis topográfico basado en datos geoespaciales, se identifican áreas clave para la implementación de estas estructuras, mejorando la infiltración del agua y reduciendo la erosión del suelo.
"""

label_resumen_proyecto = ctk.CTkLabel(frame_resumen, text=resumen_proyecto, font=("Arial", 14), justify="left", wraplength=750)
label_resumen_proyecto.pack(pady=10, padx=10)

# Sección: Objetivos Principales
frame_objetivos = ctk.CTkFrame(scrollable_frame)
frame_objetivos.pack(fill="x", padx=10, pady=10)

label_seccion_objetivos = ctk.CTkLabel(frame_objetivos, text="Objetivos Principales", font=("Arial", 20, "bold"))
label_seccion_objetivos.pack(pady=10)

objetivos_principales = """
- Regenerar los mantos freáticos y aumentar la disponibilidad de agua subterránea.
- Identificar áreas estratégicas para la construcción de zanjas de infiltración y terrazas.
- Reducir la erosión del suelo y fomentar la reforestación en zonas afectadas por la sequía.
"""

label_objetivos = ctk.CTkLabel(frame_objetivos, text=objetivos_principales, font=("Arial", 14), justify="left", wraplength=750)
label_objetivos.pack(pady=10, padx=10)

# Sección: Tecnologías Utilizadas
frame_tecnologias = ctk.CTkFrame(scrollable_frame)
frame_tecnologias.pack(fill="x", padx=10, pady=10)

label_seccion_tecnologias = ctk.CTkLabel(frame_tecnologias, text="Tecnologías Utilizadas", font=("Arial", 20, "bold"))
label_seccion_tecnologias.pack(pady=10)

tecnologias = """
El sistema utiliza tecnologías avanzadas como Python, Rasterio, Matplotlib y Plotly para procesar imágenes y generar visualizaciones interactivas. Los usuarios pueden cargar modelos digitales de elevación (GeoTIFF) para realizar análisis automatizados que recomienden el diseño óptimo de las estructuras, exportando resultados en formatos CSV para facilitar la planificación. Además, la plataforma genera gráficos en 3D y mapas interactivos que ayudan a visualizar las zonas de máximo flujo y optimización de las estructuras.
"""

label_tecnologias = ctk.CTkLabel(frame_tecnologias, text=tecnologias, font=("Arial", 14), justify="left", wraplength=750)
label_tecnologias.pack(pady=10, padx=10)

# Sección: Planteamiento del Problema
frame_problema = ctk.CTkFrame(scrollable_frame)
frame_problema.pack(fill="x", padx=10, pady=10)

label_seccion_problema = ctk.CTkLabel(frame_problema, text="Planteamiento del Problema", font=("Arial", 20, "bold"))
label_seccion_problema.pack(pady=10)

planteamiento_problema = """
La sequía constante, la erosión del suelo y la disminución de los mantos freáticos son problemas recurrentes en regiones áridas. La falta de infraestructura para captar y almacenar agua de lluvia, combinada con un crecimiento poblacional, agrava esta situación. Este proyecto busca abordar estos problemas mediante soluciones de ingeniería sostenible, que permitan regenerar los acuíferos y reducir la degradación del suelo.
"""

label_problema = ctk.CTkLabel(frame_problema, text=planteamiento_problema, font=("Arial", 14), justify="left", wraplength=750)
label_problema.pack(pady=10, padx=10)

# Sección: Resultados Esperados
frame_resultados = ctk.CTkFrame(scrollable_frame)
frame_resultados.pack(fill="x", padx=10, pady=10)

label_seccion_resultados = ctk.CTkLabel(frame_resultados, text="Resultados Esperados", font=("Arial", 20, "bold"))
label_seccion_resultados.pack(pady=10)

resultados_esperados = """
- Diseño de estructuras optimizadas como zanjas de infiltración, con coordenadas precisas.
- Exportación de resultados en formato CSV para facilitar su implementación en campo.
- Visualización de mapas topográficos, pendientes y zonas de acumulación de agua en un formato interactivo.
"""

label_resultados = ctk.CTkLabel(frame_resultados, text=resultados_esperados, font=("Arial", 14), justify="left", wraplength=750)
label_resultados.pack(pady=10, padx=10)

# Sección: Instrucciones de Uso
frame_instrucciones = ctk.CTkFrame(scrollable_frame)
frame_instrucciones.pack(fill="x", padx=10, pady=10)

label_seccion_instrucciones = ctk.CTkLabel(frame_instrucciones, text="Instrucciones de Uso", font=("Arial", 20, "bold"))
label_seccion_instrucciones.pack(pady=10)

instrucciones_uso = """
1. Cargar Imagen: Permite cargar un archivo GeoTIFF para el análisis topográfico.
2. Número de Zanjas: Introduce el número de estructuras (zanjas de infiltración) que deseas identificar.
3. Mostrar Gráficas: Genera y muestra mapas topográficos que incluyen pendientes y zonas de acumulación de agua.
4. Guardar Datos en CSV: Exporta las coordenadas y la información de las estructuras identificadas en un archivo CSV.

Este proyecto está diseñado para ser una solución efectiva y escalable para la regeneración de mantos freáticos y la mitigación de la erosión del suelo, utilizando análisis de datos geoespaciales y tecnologías de visualización avanzadas.
"""

label_instrucciones = ctk.CTkLabel(frame_instrucciones, text=instrucciones_uso, font=("Arial", 14), justify="left", wraplength=750)
label_instrucciones.pack(pady=10, padx=10)

# Ejecutar la interfaz
root.mainloop()