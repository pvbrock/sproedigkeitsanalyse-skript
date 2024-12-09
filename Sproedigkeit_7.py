import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy import trapezoid  # Import für Integration
import os
from openpyxl.drawing.image import Image
from openpyxl import load_workbook

# Pfad zur Excel-Datei
excel_file = "sproedigkeit_direkt_von_zwick_zwei_proben.xlsx"  # Passe den Pfad an deine Datei an
output_image_dir = "grafiken"  # Verzeichnis für die Bilddateien
os.makedirs(output_image_dir, exist_ok=True)  # Verzeichnis erstellen, falls es noch nicht existiert

# Excel-Datei laden und Blätter filtern
xls = pd.ExcelFile(excel_file)  # Lade die Excel-Datei
probe_sheets = [sheet for sheet in xls.sheet_names if
                sheet.startswith("Probe")]  # Nur Blätter, die mit "Probe" beginnen
print(f"Gefundene Probenblätter: {probe_sheets}")

# Ergebnisse sammeln
gesamt_ergebnisse = []

# Excel-Writer zum Speichern der Ergebnisse
with pd.ExcelWriter("Analyse_Ergebnisse.xlsx", engine="openpyxl") as writer:
    # Explizit das "Ergebnisse"-Blatt erstellen
    ergebnisse_df = pd.DataFrame(
        columns=["Probe", "Bruchpunkt (mm)", "Proportionalitätsgrenze (mm)", "Fläche 1 (N·mm)", "Fläche 2 (N·mm)",
                 "Sprödigkeitsindex (%)"])
    ergebnisse_df.to_excel(writer, sheet_name="Ergebnisse", index=False)

    # Iteriere über die gefilterten Blätter
    for sheet_name in probe_sheets:
        print(f"Verarbeite Blatt: {sheet_name}")

        # Daten aus dem aktuellen Blatt laden, Rohdaten beginnen ab Zeile 3
        data = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)  # Überspringe die ersten zwei Zeilen
        data.columns = ['mm', 'N']  # Setze die Spaltennamen

        # Deflection (mm) und Force (N)
        x = data['mm']
        y = data['N']

        # Bruchpunkt erkennen
        max_force_index = y.idxmax()  # Index des Maximums der Kraft
        max_force = y[max_force_index]  # Maximalwert der Kraft
        bruch_threshold = 0.99 * max_force  # 99 % des Maximalwerts
        bruch_indices = np.where(y[max_force_index:] < bruch_threshold)[0]
        bruch_index = max_force_index + bruch_indices[0] if len(bruch_indices) > 0 else len(y) - 1

        # Daten filtern bis zum Bruchpunkt
        x_filtered = x[:bruch_index].values
        y_filtered = y[:bruch_index].values

        # Sicherstellen, dass x_filtered monoton steigend ist
        sort_idx = np.argsort(x_filtered)
        x_filtered = x_filtered[sort_idx]
        y_filtered = y_filtered[sort_idx]

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
        prop_limit_index = prop_limit_indices[0] if len(prop_limit_indices) > 0 else -1
        prop_limit_mm = x_filtered[prop_limit_index] if prop_limit_index >= 0 else None

        # Berechnung der Fläche 1 (von 0 bis zur Proportionalitätsgrenze)
        x_for_integration = x_filtered[:prop_limit_index + 1] if prop_limit_index >= 0 else x_filtered
        y_for_integration_spline = y_smooth_spline[:prop_limit_index + 1] if prop_limit_index >= 0 else y_smooth_spline
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

        # Speichern der Grafik als PNG-Datei
        image_path = os.path.join(output_image_dir, f"{sheet_name}.png")
        plt.savefig(image_path)
        plt.close()

        # Bild in die Excel-Datei einfügen
        workbook = writer.book
        worksheet = workbook["Ergebnisse"]  # Ergebnisse-Blatt ansprechen
        img = Image(image_path)
        worksheet.add_image(img, "G5")  # Position in der Excel-Datei (z.B. G5)

    # Ergebnisse im "Ergebnisse"-Blatt speichern
    ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
    ergebnisse_df.to_excel(writer, sheet_name="Ergebnisse", index=False)

print("Analyse abgeschlossen und Ergebnisse in 'Analyse_Ergebnisse.xlsx' gespeichert.")
