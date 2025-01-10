import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy import trapezoid
from openpyxl import load_workbook
from openpyxl.drawing.image import Image


# Funktion zum Einlesen der Rohdaten aus der Excel-Datei
def read_data(excel_file):
    xls = pd.ExcelFile(excel_file)
    probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
    print(f"Gefundene Probenblätter: {probe_sheets}")
    return xls, probe_sheets


# Funktion zur Erkennung des Bruchpunkts
def detect_break_point(x, y):
    max_force_index = y.idxmax()  # Index des Maximums der Kraft
    max_force = y[max_force_index]  # Maximalwert der Kraft
    bruch_threshold = 0.99 * max_force  # 99 % des Maximalwerts
    bruch_indices = np.where(y[max_force_index:] < bruch_threshold)[0]
    if len(bruch_indices) == 0:
        print("Kein Bruchpunkt gefunden. Die Daten zeigen keine Abnahme unter 99 % des Maximums.")
        bruch_index = len(y) - 1  # Standardmäßig letztes Element wählen
    else:
        bruch_index = max_force_index + bruch_indices[0]  # Ersten Index wählen
    return bruch_index


# Funktion zum Filtern der Daten bis zum Bruchpunkt
def filter_data(x, y, bruch_index):
    x_filtered = x[:bruch_index].values
    y_filtered = y[:bruch_index].values

    # Bereinigung: Sicherstellen, dass x_filtered streng monoton steigend ist
    x_filtered, y_filtered = zip(*[(xi, yi) for xi, yi in zip(x_filtered, y_filtered) if xi > x_filtered[0]])
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

    return x_filtered, y_filtered


# Funktion zur Glättung der Daten mit Spline
def smooth_data(x_filtered, y_filtered):
    spline = UnivariateSpline(x_filtered, y_filtered)
    spline.set_smoothing_factor(1e-3)  # Glättungsfaktor erhöhen
    y_smooth_spline = spline(x_filtered)
    return spline, y_smooth_spline


# Funktion zur Berechnung der Steigung und Proportionalitätsgrenze
def calculate_slope_and_proportional_limit(x_filtered, y_filtered, spline):
    spline_derivative = spline.derivative()
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
        prop_limit_mm = None
        prop_limit_N = None

    return prop_limit_mm, prop_limit_N, spline_slopes


# Funktion zur Berechnung der Flächen
def calculate_areas(x_filtered, y_filtered, prop_limit_index, spline):
    # Fläche 1: Von 0 bis zur Proportionalitätsgrenze
    x_for_integration = x_filtered[:prop_limit_index + 1]
    y_for_integration_spline = spline(x_for_integration)
    area_1_spline = trapezoid(y_for_integration_spline, x_for_integration)

    # Fläche 2: Von der Proportionalitätsgrenze bis zum Bruchpunkt
    x_for_integration_2 = x_filtered[prop_limit_index:]
    y_for_integration_spline_2 = spline(x_for_integration_2)
    area_2_spline = trapezoid(y_for_integration_spline_2, x_for_integration_2)

    return area_1_spline, area_2_spline


# Funktion zur Berechnung des Sprödigkeitsindex
def calculate_brittleness_index(area_1_spline, area_2_spline):
    total_area = area_1_spline + area_2_spline
    sprödigkeitsindex = (area_1_spline / total_area) * 100
    return sprödigkeitsindex


# Funktion zur Visualisierung der Ergebnisse
def visualize_data(x, y, x_filtered, y_smooth_spline, bruch_index, prop_limit_mm, area_1_spline, area_2_spline,
                   sprödigkeitsindex, sheet_name):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Rohdaten', color='gray', alpha=0.5)
    plt.plot(x_filtered, y_smooth_spline, label='Spline Glättung', color='blue')
    plt.axvline(x=x[bruch_index], color='green', linestyle='--', label='Bruchpunkt')
    if prop_limit_mm is not None:
        plt.axvline(x=prop_limit_mm, color='purple', linestyle='--', label='Proportionalitätsgrenze')
    plt.fill_between(x_filtered[:bruch_index], y_smooth_spline[:bruch_index], color='cyan', alpha=0.3,
                     label=f'Fläche 1: {area_1_spline:.4f} N·mm')
    plt.fill_between(x_filtered[bruch_index:], y_smooth_spline[bruch_index:], color='orange', alpha=0.3,
                     label=f'Fläche 2: {area_2_spline:.4f} N·mm')
    plt.xlabel('Deflection (mm)')
    plt.ylabel('Force (N)')
    plt.title(f"Analyse für {sheet_name} (Sprödigkeitsindex: {sprödigkeitsindex:.2f}%)")
    plt.legend()
    plt.grid(True)

    # Speichern des Diagramms als PNG
    diagramm_datei = f"diagramm_{sheet_name}.png"
    plt.savefig(diagramm_datei)
    plt.close()


# Funktion zum Speichern der Ergebnisse in Excel
def save_results_to_excel(gesamt_ergebnisse, excel_file):
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
        ergebnisse_df.to_excel(writer, sheet_name="Ergebnisse", index=False)
    print("Analyse abgeschlossen und Ergebnisse in 'Analyse_Ergebnisse.xlsx' gespeichert.")


# Hauptfunktion zur Ausführung des gesamten Scripts
if __name__ == '__main__':
    excel_file = "sproedigkeit_direkt_von_zwick_zwei_proben.xlsx"  # Passe den Pfad an deine Datei an
    xls, probe_sheets = read_data(excel_file)

    gesamt_ergebnisse = []

    for sheet_name in probe_sheets:
        print(f"Verarbeite Blatt: {sheet_name}")

        # Daten einlesen
        data = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
        data.columns = ['mm', 'N']
        x = data['mm']
        y = data['N']

        # Bruchpunkt erkennen
        bruch_index = detect_break_point(x, y)

        # Daten filtern und bereinigen
        x_filtered, y_filtered = filter_data(x, y, bruch_index)

        # Glättung der Daten
        spline, y_smooth_spline = smooth_data(x_filtered, y_filtered)

        # Berechnung der Steigung und der Proportionalitätsgrenze
        prop_limit_mm, prop_limit_N, spline_slopes = calculate_slope_and_proportional_limit(x_filtered, y_filtered,
                                                                                            spline)

        # Berechnung der Flächen
        area_1_spline, area_2_spline = calculate_areas(x_filtered, y_filtered, prop_limit_index=0, spline=spline)

        # Berechnung des Sprödigkeitsindex
        sprödigkeitsindex = calculate_brittleness_index(area_1_spline, area_2_spline)

        # Visualisierung der Daten
        visualize_data(x, y, x_filtered, y_smooth_spline, bruch_index, prop_limit_mm, area_1_spline, area_2_spline,
                       sprödigkeitsindex, sheet_name)

        # Ergebnisse sammeln
        gesamt_ergebnisse.append({
            "Probe": sheet_name,
            "Bruchpunkt (mm)": x[bruch_index],
            "Proportionalitätsgrenze (mm)": prop_limit_mm,
            "Fläche 1 (N·mm)": area_1_spline,
            "Fläche 2 (N·mm)": area_2_spline,
            "Sprödigkeitsindex (%)": sprödigkeitsindex
        })

    # Ergebnisse speichern
    save_results_to_excel(gesamt_ergebnisse, "Analyse_Ergebnisse.xlsx")
