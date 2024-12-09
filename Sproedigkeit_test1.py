import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

# Laden der Rohdaten
data = pd.read_csv('zwick_data.csv')  # Ersetzen Sie den Dateinamen
x = data['Deflection']  # Deflexion (mm)
y = data['Force']       # Kraft (N)

# 1. Bruchpunkt erkennen
max_force_index = y.idxmax()  # Index des Maximums der Kraft
max_force = y[max_force_index]  # Maximalwert der Kraft

bruch_threshold = 0.9 * max_force  # 90 % des Maximums
bruch_index = max_force_index + np.where(y[max_force_index:] < bruch_threshold)[0][0]

# Schneiden der Daten bis zum Bruchpunkt
x_filtered = x[:bruch_index]
y_filtered = y[:bruch_index]

# Debugging: Prüfe die gefilterten Daten
print("Gefilterte Länge:", len(x_filtered), len(y_filtered))
print("Bruchindex:", bruch_index, "Deflection am Bruch:", x[bruch_index])

# Glätten der gefilterten Daten
# Spline
spline = UnivariateSpline(x_filtered, y_filtered)
spline.set_smoothing_factor(0.01)  # Weniger Glättung
y_smooth_spline = spline(x_filtered)

# Savitzky-Golay
window_length = 51
poly_order = 3
y_smooth_sg = savgol_filter(y_filtered, window_length, poly_order)

# Debugging: Prüfe die geglätteten Werte
print("Spline-Werte (erste 10):", y_smooth_spline[:10])

# 3. Visualisierung
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Rohdaten', color='gray', alpha=0.5)
plt.plot(x_filtered, y_smooth_sg, label='Savitzky-Golay Glättung', color='red')
plt.plot(x_filtered, y_smooth_spline, label='Spline Glättung', color='blue')
plt.axvline(x=x[bruch_index], color='green', linestyle='--', label='Bruchpunkt')
plt.xlabel('Deflexion (mm)')
plt.ylabel('Kraft (N)')
plt.title('Datenanalyse: Bruchpunkt und Glättung')
plt.legend()
plt.grid(True)
plt.show()
