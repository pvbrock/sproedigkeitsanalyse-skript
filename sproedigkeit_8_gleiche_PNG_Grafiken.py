import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy import trapezoid  # Import für Integration
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import io

# Pfad zur Excel-Datei
excel_file = "sproedigkeit_direkt_von_zwick_zwei_proben.xlsx"  # Passe den Pfad an deine Datei an
output_file = "Analyse_Ergebnisse.xlsx"

# Excel-Datei laden und Blätter filtern
xls = pd.ExcelFile(excel_file)
probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
print(f"Gefundene Probenblätter: {probe_sheets}")

# Ergebnisse sammeln
gesamt_ergebnisse = []

# Schritt 1: Analyse durchführen und Ergebnisse speichern
with pd.ExcelWriter(output_file) as writer:
    for sheet_name in probe_sheets:
        print(f"Verarbeite Blatt: {sheet_name}")

        # Daten laden
        data = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
        data.columns = ['mm', 'N']

        # Deflection (mm) und Force (N)
        x = data['mm']
        y = data['N']

        # Bruchpunkt berechnen
        max_force_index = y.idxmax()
        bruch_threshold = 0.99 * y[max_force_index]
        bruch_indices = np.where(y[max_force_index:] < bruch_threshold)[0]
        bruch_index = max_force_index + bruch_indices[0] if len(bruch_indices) > 0 else len(y) - 1

        x_filtered = x[:bruch_index].values
        y_filtered = y[:bruch_index].values

        # Sicherstellen, dass x_filtered streng monoton steigend ist
        x_filtered, y_filtered = zip(*[(xi, yi) for xi, yi in zip(x_filtered, y_filtered) if xi > x_filtered[0]])
        x_filtered = np.array(x_filtered)
        y_filtered = np.array(y_filtered)

        # Glättung mit Univariate Spline
        spline = UnivariateSpline(x_filtered, y_filtered)
        spline.set_smoothing_factor(10000000)
        y_smooth_spline = spline(x_filtered)

        # Berechnung der ersten Ableitung der Spline-Kurve
        spline_derivative = spline.derivative()
        spline_slopes = spline_derivative(x_filtered)

        # Proportionalitätsgrenze bestimmen
        initial_slope = np.mean(spline_slopes[:10])
        threshold = 0.99 * initial_slope
        prop_limit_indices = np.where(spline_slopes < threshold)[0]
        if len(prop_limit_indices) > 0:
            prop_limit_index = prop_limit_indices[0]
            prop_limit_mm = x_filtered[prop_limit_index]
            prop_limit_N = y_filtered[prop_limit_index]
        else:
            prop_limit_index = None
            prop_limit_mm = None
            prop_limit_N = None

        # Integrale berechnen
        if prop_limit_index is not None:
            area_1 = trapezoid(y_smooth_spline[:prop_limit_index + 1], x_filtered[:prop_limit_index + 1])
            area_2 = trapezoid(y_smooth_spline[prop_limit_index:], x_filtered[prop_limit_index:])
        else:
            area_1 = 0
            area_2 = trapezoid(y_smooth_spline, x_filtered)

        # Sprödigkeitsindex berechnen
        total_area = area_1 + area_2
        sprödigkeitsindex = (area_1 / total_area) * 100 if total_area > 0 else 0

        # Ergebnisse sammeln
        gesamt_ergebnisse.append({
            "Probe": sheet_name,
            "Bruchpunkt (mm)": x[bruch_index],
            "Proportionalitätsgrenze (mm)": prop_limit_mm,
            "Fläche 1 (N·mm)": area_1,
            "Fläche 2 (N·mm)": area_2,
            "Sprödigkeitsindex (%)": sprödigkeitsindex
        })

        # Rohdaten und Spline speichern
        results_df = pd.DataFrame({
            'mm': x_filtered,
            'N': y_filtered,
            'Spline': y_smooth_spline
        })
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Ergebnisse speichern
    ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
    ergebnisse_df.to_excel(writer, sheet_name="Ergebnisse", index=False)

# Schritt 2: Diagramme hinzufügen
wb = load_workbook(output_file)
for sheet_name in probe_sheets:
    ws = wb[sheet_name]

    # Diagramm erstellen
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Rohdaten', color='gray', alpha=0.5)
    plt.plot(x_filtered, y_smooth_spline, label='Spline Glättung', color='blue')
    plt.axvline(x=x[bruch_index], color='green', linestyle='--', label='Bruchpunkt')

    if prop_limit_mm is not None:
        plt.axvline(x=prop_limit_mm, color='red', linestyle='--', label='Proportionalitätsgrenze')
        plt.fill_between(x_filtered[:prop_limit_index + 1], 0, y_smooth_spline[:prop_limit_index + 1], color='green', alpha=0.2, label='Fläche 1')
        plt.fill_between(x_filtered[prop_limit_index:], 0, y_smooth_spline[prop_limit_index:], color='orange', alpha=0.2, label='Fläche 2')

    plt.xlabel('Deflection (mm)')
    plt.ylabel('Force (N)')
    plt.title(f"Analyse für {sheet_name}")
    plt.legend()
    plt.grid(True)

    # Bild in Byte-Stream speichern
    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    plt.close()

    # Bild hinzufügen
    img = Image(img_stream)
    img.anchor = 'H1'
    ws.add_image(img)

# Excel-Datei speichern
wb.save(output_file)
print("Analyse abgeschlossen. Ergebnisse und Diagramme in 'Analyse_Ergebnisse.xlsx' gespeichert.")
