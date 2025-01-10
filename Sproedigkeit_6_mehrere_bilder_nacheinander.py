import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy import trapezoid

# Pfad zur Excel-Datei
excel_file = "sproedigkeit_direkt_von_zwick_zwei_proben.xlsx"

# Excel-Datei laden und Blätter filtern
xls = pd.ExcelFile(excel_file)
probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
print(f"Gefundene Probenblätter: {probe_sheets}")

# Ergebnisse sammeln
gesamt_ergebnisse = []

# Iteriere über die gefilterten Blätter
for sheet_name in probe_sheets:
    print(f"Verarbeite Blatt: {sheet_name}")

    # Daten aus dem aktuellen Blatt laden, Rohdaten beginnen ab Zeile 3
    data = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)  # Überspringe die ersten zwei Zeilen
    data.columns = ['mm', 'N']

    # Überprüfen der ersten Zeilen der Daten
    print(data.head())

    # Deflection (mm) und Force (N)
    x = data['mm']
    y = data['N']

    # Bruchpunkt erkennen
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

    from scipy.interpolate import UnivariateSpline

    # Daten filtern bis zum Bruchpunkt
    x_filtered = x[:bruch_index].values
    y_filtered = y[:bruch_index].values

    # Daten filtern bis zum Bruchpunkt
    x_filtered_sorted, y_filtered_sorted = zip(*sorted(zip(x_filtered, y_filtered)))

    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

    # Glättungsfaktor erhöhen, z.B. auf 1e-3
    spline = UnivariateSpline(x_filtered_sorted, y_filtered_sorted)
    spline.set_smoothing_factor(10000)  # Erhöhe den Glättungsfaktor, wenn die Spline-Berechnung fehlschlägt
    y_smooth_spline = spline(x_filtered_sorted)

    # Sortieren nach x, um sicherzustellen, dass die Spline-Berechnung funktioniert
    sorted_data = sorted(zip(x_filtered, y_filtered))
    x_filtered_sorted, y_filtered_sorted = zip(*sorted_data)

    # Glättung der Daten: Univariate Spline mit höherem Glättungsfaktor
    spline = UnivariateSpline(x_filtered_sorted, y_filtered_sorted)
    spline.set_smoothing_factor(1e-5)  # Erhöht den Glättungsfaktor
    y_smooth_spline = spline(x_filtered_sorted)

    # Berechnung der ersten Ableitung der Spline-Kurve
    spline_derivative = spline.derivative()
    spline_slopes = spline_derivative(x_filtered_sorted)

    # Weitere Berechnungen (Proportionalitätsgrenze, Flächen usw.)

    gesamt_ergebnisse.append({
        "Probe": sheet_name,
        "Bruchpunkt (mm)": x[bruch_index],
        "Fläche 1 (N·mm)": 0,  # Placeholder für die Fläche
        "Fläche 2 (N·mm)": 0,  # Placeholder für die Fläche
        "Sprödigkeitsindex (%)": 0  # Placeholder für den Index
    })

# Ergebnisse in Excel speichern
with pd.ExcelWriter("Analyse_Ergebnisse.xlsx", engine='openpyxl') as writer:
    ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
    ergebnisse_df.to_excel(writer, sheet_name="Ergebnisse", index=False)

    # Probenblätter hinzufügen und visualisieren
    for sheet_name in probe_sheets:
        # Beispiel-Daten zur Probe (kann beliebig erweitert werden)
        data_to_save = data[['mm', 'N']]  # Daten für die Probe
        data_to_save.to_excel(writer, sheet_name=sheet_name, index=False)

print("Analyse abgeschlossen und Ergebnisse in 'Analyse_Ergebnisse.xlsx' gespeichert.")
