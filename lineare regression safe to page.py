import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapezoid
from sklearn.linear_model import LinearRegression
from openpyxl import load_workbook

# Path to data file (can be .xlsx, .xls, or .csv)
data_file = "zwick_alle_Daten_alle_chargen.xls"  # Change to your file path

# Check file type and load accordingly
if data_file.endswith(".xlsx"):
    xls = pd.ExcelFile(data_file)
    probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
elif data_file.endswith(".xls"):
    xls = pd.ExcelFile(data_file, engine="xlrd")
    probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
elif data_file.endswith(".csv"):
    csv_data = pd.read_csv(data_file)
    probe_sheets = ["CSV_Data"]
else:
    raise ValueError(f"Unsupported file format: {data_file}")

print(f"Gefundene Probenblätter: {probe_sheets}")

# Ergebnisse sammeln
gesamt_ergebnisse = []

# Stützweite (L), Breite (b), Höhe (h) -> Falls bekannt, hier Werte eintragen!
L = 300  # Beispielwert in mm
b = 20   # Beispielwert in mm
h = 20   # Beispielwert in mm

# Iteriere über die gefilterten Blätter
for sheet_name in probe_sheets:
    print(f"\n=== Verarbeite Blatt: {sheet_name} ===")

    # Daten aus dem aktuellen Blatt laden (Rohdaten beginnen ab Zeile 3)
    data = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
    data.columns = ['mm', 'N']  # Spalten umbenennen

    # Deflection (mm) und Force (N)
    x = data['mm']
    y = data['N']

    # -------------------------------------------------------------
    # 1) Ermittlung des Punktes mit der größten Kraft
    # -------------------------------------------------------------
    max_force_index = y.idxmax()
    max_force = y[max_force_index]

    # Daten für Analyse filtern
    x_filtered = x[:max_force_index + 1].values
    y_filtered = y[:max_force_index + 1].values

    # Auf streng steigende x filtern (Dubletten entfernen)
    x_filtered, y_filtered = zip(*[
        (xi, yi) for xi, yi in zip(x_filtered, y_filtered)
        if xi > x_filtered[0]
    ])
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

    # -------------------------------------------------------------
    # 2) Spline-Glättung
    # -------------------------------------------------------------
    spline_smooth_factor = 1e12
    spline = UnivariateSpline(x_filtered, y_filtered)
    spline.set_smoothing_factor(spline_smooth_factor)
    y_smooth_spline = spline(x_filtered)

    # -------------------------------------------------------------
    # 3) Berechnung des MSE für die Spline-Glättung
    # -------------------------------------------------------------
    mse_spline = np.mean((y_filtered - y_smooth_spline) ** 2)
    print(f"MSE für Spline-Glättung ({sheet_name}): {mse_spline:.6f}")

    # -------------------------------------------------------------
    # 4) Lineare Regression im Bereich 0.5 mm bis 3.5 mm
    # -------------------------------------------------------------
    regression_start_point_mm = 0.5
    regression_end_point_mm = 3

    regression_mask = (x_filtered >= regression_start_point_mm) & (x_filtered <= regression_end_point_mm)
    x_regression = x_filtered[regression_mask].reshape(-1, 1)
    y_regression = y_filtered[regression_mask]

    linear_model = LinearRegression()
    linear_model.fit(x_regression, y_regression)

    slope = linear_model.coef_[0]  # Steigung (N/mm)
    intercept = linear_model.intercept_
    r_squared = linear_model.score(x_regression, y_regression)
    r_value = np.sqrt(r_squared)  # Berechnung des Korrelationskoeffizienten R

    print(f"R-Wert der linearen Regression für {sheet_name}: {r_value:.4f}")

    # -------------------------------------------------------------
    # 5) Berechnung des Elastizitätsmoduls (E-Modul)
    # -------------------------------------------------------------
    if L > 0 and b > 0 and h > 0:
        E_modul = (L ** 3 * slope) / (4 * b * h ** 3)
    else:
        E_modul = None  # Falls L, b oder h nicht definiert sind

    print(f"E-Modul für {sheet_name}: {E_modul:.2f} N/mm²")

    # -------------------------------------------------------------
    # 6) Berechnung der Flächen und Sprödigkeitsindex
    # -------------------------------------------------------------
    close_threshold = 0.01
    deviations_bw = np.abs(y_smooth_spline - linear_model.predict(x_filtered.reshape(-1, 1)))
    deviation_ratios_bw = deviations_bw / y_smooth_spline
    mask_bw_close = (deviation_ratios_bw < close_threshold)
    close_indices_bw = np.where(mask_bw_close)[0]

    if len(close_indices_bw) > 0:
        first_close_index = close_indices_bw[0]
        close_x = x_filtered[first_close_index]
    else:
        close_x = None

    if close_x is not None:
        first_close_index = np.where(x_filtered >= close_x)[0][0]
        area_1 = trapezoid(y_smooth_spline[:first_close_index + 1], x_filtered[:first_close_index + 1])
        area_2 = trapezoid(y_smooth_spline[first_close_index:], x_filtered[first_close_index:])
    else:
        area_1 = 0
        area_2 = trapezoid(y_smooth_spline, x_filtered)

    total_area = area_1 + area_2
    sprödigkeitsindex = (area_1 / total_area) * 100 if total_area > 0 else 0

    # Ergebnisse speichern
    gesamt_ergebnisse.append({
        "Probe": sheet_name,
        "Max. Kraft (N)": max_force,
        "Max. Kraft (mm)": x[max_force_index],
        "Regression Steigung (N/mm)": slope,
        "Regression Bestimmtheitsmaß (R²)": r_squared,
        "Regression Korrelationskoeffizient (R)": r_value,
        "Elastizitätsmodul (N/mm²)": E_modul,
        "MSE Spline": mse_spline,
        "Annäherung (mm)": close_x,
        "Fläche 1 (N*mm)": area_1,
        "Fläche 2 (N*mm)": area_2,
        "Sprödigkeitsindex (%)": sprödigkeitsindex
    })

# -------------------------------------------------------------
# 7) Ergebnisse in Excel speichern
# -------------------------------------------------------------
output_file = "Analyse_Ergebnisse_Lineare_Regression_alle_Chargen.xlsx"
new_sheet_name = f"sm({spline_smooth_factor:.1e})_sh({close_threshold})_re({regression_start_point_mm}-{regression_end_point_mm})"

ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)

try:
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a") as writer:
        ergebnisse_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    print(f"Ergebnisse erfolgreich gespeichert in '{output_file}' (Tabelle: '{new_sheet_name}').")
except FileNotFoundError:
    print(f"Datei '{output_file}' existiert nicht. Erstelle neue Datei.")
    ergebnisse_df.to_excel(output_file, sheet_name=new_sheet_name, index=False)
