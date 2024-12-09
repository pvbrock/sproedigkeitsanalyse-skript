import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy import trapezoid  # Import für Integration

# Pfad zur Excel-Datei
excel_file = "sproedigkeit_direkt_von_zwick_zwei_proben.xlsx"  # Passe den Pfad an deine Datei an

# Excel-Datei laden und Blätter filtern
xls = pd.ExcelFile(excel_file)  # Lade die Excel-Datei
probe_sheets = [sheet for sheet in xls.sheet_names if
                sheet.startswith("Probe")]  # Nur Blätter, die mit "Probe" beginnen
print(f"Gefundene Probenblätter: {probe_sheets}")

# Ergebnisse sammeln
gesamt_ergebnisse = []

# Iteriere über die gefilterten Blätter
for sheet_name in probe_sheets:
    print(f"Verarbeite Blatt: {sheet_name}")

    # Daten aus dem aktuellen Blatt laden, Rohdaten beginnen ab Zeile 3
    data = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)  # Überspringe die ersten zwei Zeilen
    data.columns = ['mm', 'N']  # Setze die Spaltennamen

    # Überprüfen der ersten Zeilen der Daten
    print(data.head())

    # Deflection (mm) und Force (N)
    x = data['mm']
    y = data['N']

    # Bruchpunkt erkennen
    print("Berechnung des Bruchpunkts gestartet...")
    max_force_index = y.idxmax()  # Index des Maximums der Kraft
    max_force = y[max_force_index]  # Maximalwert der Kraft
    bruch_threshold = 0.99 * max_force  # 99 % des Maximalwerts

    bruch_indices = np.where(y[max_force_index:] < bruch_threshold)[0]
    if len(bruch_indices) == 0:
        print("Kein Bruchpunkt gefunden. Die Daten zeigen keine Abnahme unter 99 % des Maximums.")
        bruch_index = len(y) - 1  # Standardmäßig letztes Element wählen
    else:
        bruch_index = max_force_index + bruch_indices[0]  # Ersten Index wählen
        print(f"Bruchpunkt gefunden bei Index {bruch_index} mit Deflection {x[bruch_index]} mm.")

    # Daten filtern bis zum Bruchpunkt
    x_filtered = x[:bruch_index].values
    y_filtered = y[:bruch_index].values

    # Bereinigung: Sicherstellen, dass x_filtered streng monoton steigend ist
    x_filtered, y_filtered = zip(*[(xi, yi) for xi, yi in zip(x_filtered, y_filtered) if xi > x_filtered[0]])
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

    print("Gefilterte und bereinigte Datenlänge:", len(x_filtered), len(y_filtered))

    # Glättung der Daten: Univariate Spline
    spline = UnivariateSpline(x_filtered, y_filtered)
    spline.set_smoothing_factor(10000000)
    y_smooth_spline = spline(x_filtered)

    # Berechnung der ersten Ableitung der Spline-Kurve
    spline_derivative = spline.derivative()

    # Steigung an jedem Punkt berechnen
    spline_slopes = spline_derivative(x_filtered)

    # Initiale Steigung bestimmen
    initial_slope = np.mean(spline_slopes[:10])  # Mittelwert der ersten 10 Werte
    threshold = 0.99 * initial_slope  # 1% Abweichung als Kriterium

    # Finden des Indexes, an dem die Steigung unter den Schwellenwert fällt
    prop_limit_indices = np.where(spline_slopes < threshold)[0]

    if len(prop_limit_indices) > 0:
        prop_limit_index = prop_limit_indices[0]
        prop_limit_mm = x_filtered[prop_limit_index]
        prop_limit_N = y_filtered[prop_limit_index]
        print(f"Proportionalitätsgrenze gefunden bei: {prop_limit_mm:.4f} mm, {prop_limit_N:.4f} N")
    else:
        print("Keine Proportionalitätsgrenze gefunden. Die Kurve bleibt linear oder keine Abweichung erkannt.")

    # Berechnung der Fläche 1 (von 0 bis zur Proportionalitätsgrenze)
    x_for_integration = x_filtered[:prop_limit_index + 1]
    y_for_integration_spline = y_smooth_spline[:prop_limit_index + 1]
    area_1_spline = trapezoid(y_for_integration_spline, x_for_integration)

    # Berechnung der Fläche 2 (von der Proportionalitätsgrenze bis zum Bruchpunkt)
    x_for_integration_2 = x_filtered[prop_limit_index:bruch_index + 1]
    y_for_integration_spline_2 = y_smooth_spline[prop_limit_index:bruch_index + 1]
    area_2_spline = trapezoid(y_for_integration_spline_2, x_for_integration_2)

    # Berechnung des Sprödigkeitsindex
    total_area = area_1_spline + area_2_spline
    sprödigkeitsindex = (area_1_spline / total_area) * 100

    # Ergebnisse sammeln
    gesamt_ergebnisse.append({
        "Probe": sheet_name,
        "Bruchpunkt (mm)": x[bruch_index],
        "Proportionalitätsgrenze (mm)": prop_limit_mm,
        "Fläche 1 (N·mm)": area_1_spline,
        "Fläche 2 (N·mm)": area_2_spline,
        "Sprödigkeitsindex (%)": sprödigkeitsindex
    })

    # Visualisierung
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Rohdaten', color='gray', alpha=0.5)
    plt.plot(x_filtered, y_smooth_spline, label='Spline Glättung', color='blue')
    plt.axvline(x=x[bruch_index], color='green', linestyle='--', label='Bruchpunkt')
    plt.axvline(x=prop_limit_mm, color='purple', linestyle='--', label='Proportionalitätsgrenze')
    plt.fill_between(x_for_integration, y_for_integration_spline, color='cyan', alpha=0.3,
                     label=f'Fläche 1: {area_1_spline:.4f} N·mm')
    plt.fill_between(x_for_integration_2, y_for_integration_spline_2, color='orange', alpha=0.3,
                     label=f'Fläche 2: {area_2_spline:.4f} N·mm')
    plt.xlabel('Deflection (mm)')
    plt.ylabel('Force (N)')
    plt.title(f"Analyse für {sheet_name} (Sprödigkeitsindex: {sprödigkeitsindex:.2f}%)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Ergebnisse in Excel speichern
ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
ergebnisse_df.to_excel("Analyse_Ergebnisse.xlsx", index=False)
with pd.ExcelWriter("Analyse_Ergebnisse.xlsx") as writer:
    ergebnisse_df.to_excel(writer, sheet_name="Ergebnisse", index=False)
print("Analyse abgeschlossen und Ergebnisse in 'Analyse_Ergebnisse.xlsx' gespeichert.")
