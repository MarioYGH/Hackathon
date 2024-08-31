import numpy as np
from scipy.optimize import minimize

# Parámetros adicionales
P_avg = 0.01  # Precipitación media en m/s (equivale a 36 mm/h)
T = 25  # Temperatura en grados Celsius
RH = 60  # Humedad relativa en %


# Definir parámetros físicos y funciones relevantes (conductividad hidráulica, etc.)
def K(theta):
    # Relación entre saturación y conductividad hidráulica (modelo simplificado)
    K_sat = 1e-6  # Conductividad hidráulica saturada en m/s (típico para suelos arcillosos)
    return K_sat * theta ** 2  # Ejemplo simple, conductividad cuadrática con saturación


def E(theta, T, RH):
    # Relación para la evaporación (simplificada)
    # Evaporación típica ajustada según temperatura y humedad relativa
    E_max = 5e-7  # Máxima tasa de evaporación en m/s (0.5 mm/h)
    return E_max * (1 - theta) * (T / 25) * (RH / 100)


def infiltration_area(scaling_factor, P_avg, Q_in):
    # Dimensiones proporcionales
    w = scaling_factor
    d = 2 * scaling_factor
    l = 3 * scaling_factor

    A = w * l
    V = w * l * d

    # Suponer una saturación inicial (theta inicial)
    theta = 0.3  # Saturación inicial del suelo, típica para suelos parcialmente secos
    grad_psi_z = -1  # Supuesto gradiente matricial simple

    # Calcular la tasa de infiltración considerando el flujo volumétrico que llega a la zanja
    infiltration = (K(theta) * grad_psi_z + (P_avg + Q_in) / A) - E(theta, T, RH)

    # Multiplicar por el área para obtener la infiltración total
    return infiltration * A


# Función objetivo para optimización (maximizar la infiltración)
def objective(scaling_factor, Q_in):
    return -infiltration_area(scaling_factor, P_avg, Q_in)  # Usamos el negativo para maximizar


# Función principal que optimiza las dimensiones de la zanja
def optimize_trench(Q_in):
    # Restricciones geométricas y físicas
    V_min = 1.0  # Volumen mínimo en m³
    A_max = 10.0  # Área máxima en m²

    cons = [{'type': 'ineq', 'fun': lambda s: s * (2 * s) * (3 * s) - V_min},  # Volumen mínimo
            {'type': 'ineq', 'fun': lambda s: A_max - s * (3 * s)}]  # Área máxima

    # Parámetros iniciales (suponer un punto de partida)
    initial_guess = [0.5]  # Escalamiento inicial en metros

    # Optimización
    result = minimize(objective, initial_guess, args=(Q_in,), constraints=cons, method='SLSQP')
    optimal_scaling_factor = result.x[0]

    # Dimensiones óptimas basadas en el factor de escalamiento
    optimal_w = optimal_scaling_factor
    optimal_d = 2 * optimal_scaling_factor
    optimal_l = 3 * optimal_scaling_factor

    # Calcular el volumen óptimo
    optimal_volume = optimal_w * optimal_d * optimal_l

    # Devolver los resultados
    return {
        'optimal_w': optimal_w,
        'optimal_d': optimal_d,
        'optimal_l': optimal_l,
        'optimal_volume': optimal_volume,
        'maximum_infiltration': -result.fun
    }

# Ejemplo de uso
Q_in = 0.001  # Flujo volumétrico en m³/s (equivale a 1 litro por segundo)
results = optimize_trench(Q_in)

print(f"Optimal dimensions: w = {results['optimal_w']:.2f} m, d = {results['optimal_d']:.2f} m, l = {results['optimal_l']:.2f} m")
print(f"Optimal volume: V = {results['optimal_volume']:.2f} m³")
print(f"Maximum infiltration: {results['maximum_infiltration']:.4f} m³/s")
