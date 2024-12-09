import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy import trapezoid
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

# Pfad zur Excel-Datei
excel_file = "sproedigkeit_direkt_von_zwick_zwei_proben.xlsx"
output_file = "Analyse_Ergebnisse.xlsx"

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

    # Daten filtern bis zum Bruchpunkt
    x_filtered = x[:bruch_index]
    y_filtered = y[:bruch_index]

    # Glättung der Daten: Univariate Spline
    spline = UnivariateSpline(x_filtered, y_filtered)
    spline.set_smoothing_factor(1e-5)  # Glättungsfaktor einstellen
    y_smooth_spline = spline(x_filtered)

    # Berechnung der ersten Ableitung der Spline-Kurve
    spline_derivative = spline.derivative()
    spline_slopes = spline_derivative(x_filtered)

    # Ergebnisse sammeln
    gesamt_ergebnisse.append({
        "Probe": sheet_name,
        "Bruchpunkt (mm)": x[bruch_index],
        "Fläche 1 (N·mm)": 0,  # Placeholder für die Fläche
        "Fläche 2 (N·mm)": 0,  # Placeholder für die Fläche
        "Sprödigkeitsindex (%)": 0  # Placeholder für den Index
    })

    # Schritt 1: Diagramm erstellen und speichern
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Rohdaten', color='gray', alpha=0.5)
    plt.plot(x_filtered, y_smooth_spline, label='Spline Glättung', color='blue')
    plt.axvline(x=x[bruch_index], color='green', linestyle='--', label='Bruchpunkt')

    plt.xlabel('Deflection (mm)')
    plt.ylabel('Force (N)')
    plt.title(f"Analyse für {sheet_name}")
    plt.legend()
    plt.grid(True)

    # Speichern des Diagramms als PNG
    diagramm_datei = f"diagramm_{sheet_name}.png"
    plt.savefig(diagramm_datei)
    plt.close()

    # Schritt 2: Diagramm in Excel-Blatt einfügen
    wb = load_workbook(output_file)
    ws = wb[sheet_name]

    img = Image(diagramm_datei)
    img.anchor = 'H1'  # Position der Grafik im Excel-Blatt
    ws.add_image(img)

    # Excel-Datei speichern
    wb.save(output_file)

# Ergebnisse in Excel speichern
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
    ergebnisse_df.to_excel(writer, sheet_name="Ergebnisse", index=False)

print("Analyse abgeschlossen und Ergebnisse in 'Analyse_Ergebnisse.xlsx' gespeichert.")
