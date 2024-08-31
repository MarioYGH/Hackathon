import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Parámetros del modelo de Horton
f_0 = 30  # mm/h (tasa inicial de infiltración)
f_c = 5   # mm/h (tasa constante de infiltración)
k = 0.5   # 1/h (coeficiente de decaimiento)

# Parámetros de evaporación
E = 2  # mm/h (tasa constante de evaporación)

# Función de infiltración (Horton)
def infiltration_rate(t):
    return f_c + (f_0 - f_c) * np.exp(-k * t)

# Función de evaporación (constante)
def evaporation_rate(t):
    return E

# Tiempo total de simulación (horas)
t_final = 10

# Calcular la cantidad de agua infiltrada y evaporada
time_points = np.linspace(0, t_final, 100)
infiltration_total = np.array([quad(infiltration_rate, 0, t)[0] for t in time_points])
evaporation_total = np.array([quad(evaporation_rate, 0, t)[0] for t in time_points])

# Agua infiltrada considerando evaporación
water_infiltrated = infiltration_total - evaporation_total

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(time_points, infiltration_total, label='Infiltración Total (mm)')
plt.plot(time_points, evaporation_total, label='Evaporación Total (mm)')
plt.plot(time_points, water_infiltrated, label='Agua Infiltrada (mm)', linestyle='--', color='green')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Cantidad de Agua (mm)')
plt.title('Infiltración y Evaporación del Agua en la Zanja')
plt.legend()
plt.grid(True)
plt.show()
