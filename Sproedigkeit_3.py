import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import pandas as pd

# Laden der Excel-Daten
excel_file = "zwick_data.xlsx"  # Pfad zur Excel-Datei
data = pd.read_excel(excel_file)  # Annahme: Spalten heißen "mm" und "N"
print("Daten aus Excel erfolgreich geladen.")

# Überprüfen der ersten Zeilen der Daten
print(data.head())

# Deflection (mm) und Force (N)
x = data['mm']
y = data['N']

# 1. Bruchpunkt erkennen
print("Berechnung des Bruchpunkts gestartet...")
max_force_index = y.idxmax()  # Index des Maximums der Kraft
max_force = y[max_force_index]  # Maximalwert der Kraft
bruch_threshold = 0.99 * max_force  # 90 % des Maximalwerts

bruch_indices = np.where(y[max_force_index:] < bruch_threshold)[0]
if len(bruch_indices) == 0:
    print("Kein Bruchpunkt gefunden. Die Daten zeigen keine Abnahme unter 90 % des Maximums.")
    bruch_index = len(y) - 1  # Standardmäßig letztes Element wählen
else:
    bruch_index = max_force_index + bruch_indices[0]  # Ersten Index wählen
    print(f"Bruchpunkt gefunden bei Index {bruch_index} mit Deflection {x[bruch_index]} mm.")

# 2. Daten filtern bis zum Bruchpunkt
x_filtered = x[:bruch_index].values
y_filtered = y[:bruch_index].values

# Bereinigung: Sicherstellen, dass x_filtered streng monoton steigend ist
x_filtered, y_filtered = zip(*[(xi, yi) for xi, yi in zip(x_filtered, y_filtered) if xi > x_filtered[0]])
x_filtered = np.array(x_filtered)
y_filtered = np.array(y_filtered)

print("Gefilterte und bereinigte Datenlänge:", len(x_filtered), len(y_filtered))

# 3. Glättung der Daten
# Savitzky-Golay-Filter
window_length = 51*20  # Muss ungerade sein
poly_order = 3
y_smooth_sg = savgol_filter(y_filtered, window_length, poly_order)

# Univariate Spline
spline = UnivariateSpline(x_filtered, y_filtered)
spline.set_smoothing_factor(10000000)
y_smooth_spline = spline(x_filtered)

# Berechnung der ersten Ableitung der Spline-Kurve
spline_derivative = spline.derivative()

# Steigung an jedem Punkt berechnen
spline_slopes = spline_derivative(x_filtered)

# Initiale Steigung bestimmen (Durchschnitt der ersten Werte)
initial_slope = np.mean(spline_slopes[:10])  # Mittelwert der ersten 10 Werte
threshold = 0.99 * initial_slope  # 5% Abweichung als Kriterium

# Finden des Indexes, an dem die Steigung unter den Schwellenwert fällt
prop_limit_indices = np.where(spline_slopes < threshold)[0]

if len(prop_limit_indices) > 0:
    prop_limit_index = prop_limit_indices[0]
    prop_limit_mm = x_filtered[prop_limit_index]
    prop_limit_N = y_filtered[prop_limit_index]
    print(f"Proportionalitätsgrenze gefunden bei: {prop_limit_mm:.4f} mm, {prop_limit_N:.4f} N")
else:
    print("Keine Proportionalitätsgrenze gefunden. Die Kurve bleibt linear oder keine Abweichung erkannt.")

from numpy import trapezoid  # Import von numpy.trapezoid

# Finde den Index der Proportionalitätsgrenze
prop_limit_index = np.where(x_filtered >= prop_limit_mm)[0][0]

# Berechne die Werte der geglätteten Spline-Kurve bis zur Proportionalitätsgrenze
x_for_integration = x_filtered[:prop_limit_index+1]  # x-Werte von 0 mm bis zur Proportionalitätsgrenze
y_for_integration_spline = y_smooth_spline[:prop_limit_index+1]  # Geglättete Spline-Werte

# Numerische Integration der geglätteten Spline-Kurve
area_1_spline = trapezoid(y_for_integration_spline, x_for_integration)

print(f"Fläche 1 (unter der geglätteten Spline-Kurve von 0 mm bis Proportionalitätsgrenze): {area_1_spline:.4f} N·mm")

# Bereich von der Proportionalitätsgrenze bis zum Bruchpunkt
x_for_integration_2 = x_filtered[prop_limit_index:bruch_index+1]  # x-Werte von Proportionalitätsgrenze bis Bruchpunkt
y_for_integration_spline_2 = y_smooth_spline[prop_limit_index:bruch_index+1]  # Spline-Werte im selben Bereich

# Numerische Integration über diesen Bereich
area_2_spline = trapezoid(y_for_integration_spline_2, x_for_integration_2)

print(f"Fläche 2 (unter der Spline-Kurve von Proportionalitätsgrenze bis Bruchpunkt): {area_2_spline:.4f} N·mm")

# Berechnung des Sprödigkeitsindex
total_area = area_1_spline + area_2_spline  # Gesamtfläche
sprödigkeitsindex = (area_1_spline / total_area) * 100  # In Prozent

print(f"Sprödigkeitsindex: {sprödigkeitsindex:.2f}%")

# Visualisierung mit Sprödigkeitsindex
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Rohdaten', color='gray', alpha=0.5)
#plt.plot(x_filtered, y_smooth_sg, label='Savitzky-Golay Glättung', color='red')
plt.plot(x_filtered, y_smooth_spline, label='Spline Glättung', color='blue')
plt.axvline(x=x[bruch_index], color='green', linestyle='--', label='Bruchpunkt')
if len(prop_limit_indices) > 0:
    plt.axvline(x=prop_limit_mm, color='purple', linestyle='--', label='Proportionalitätsgrenze')

# Färben des Bereichs für Fläche 1 (0 mm bis Proportionalitätsgrenze)
plt.fill_between(x_for_integration, y_for_integration_spline, color='cyan', alpha=0.3, label=f'Fläche 1 (Spline): {area_1_spline:.4f} N·mm')

# Färben des Bereichs für Fläche 2 (Proportionalitätsgrenze bis Bruchpunkt)
plt.fill_between(x_for_integration_2, y_for_integration_spline_2, color='orange', alpha=0.3, label=f'Fläche 2 (Spline): {area_2_spline:.4f} N·mm')

plt.xlabel('Deflection (mm)')
plt.ylabel('Force (N)')
plt.title(f'Datenanalyse: Flächen und Sprödigkeitsindex ({sprödigkeitsindex:.2f}%)')
plt.legend()
plt.grid(True)
plt.show()



