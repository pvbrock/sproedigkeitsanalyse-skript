import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from sklearn.linear_model import LinearRegression

# Konfiguration von Parametern
RAW_DATA_FILE_PATH = r"rohdaten_beispiel.xls"
OUTPUT_FILE_EXCEL = r"output\analyse_ergebnisse.xlsx"
OUTPUT_PLOT_DIRECTORY = r"output\plots"
SHOW_PLOTS = True

# Erstelle output Ordner
folder = os.path.dirname(OUTPUT_FILE_EXCEL)
if folder:
    os.makedirs(folder, exist_ok=True)
os.makedirs(OUTPUT_PLOT_DIRECTORY, exist_ok=True)

# Überprüfen des Dateiformats und entsprechendes Laden
if RAW_DATA_FILE_PATH.endswith(".xlsx"):
    xls = pd.ExcelFile(RAW_DATA_FILE_PATH)
    probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
elif RAW_DATA_FILE_PATH.endswith(".xls"):
    xls = pd.ExcelFile(RAW_DATA_FILE_PATH, engine="xlrd")
    probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
elif RAW_DATA_FILE_PATH.endswith(".csv"):
    csv_data = pd.read_csv(RAW_DATA_FILE_PATH)
    probe_sheets = ["CSV_Data"]
else:
    raise ValueError(f"Nicht unterstütztes Dateiformat: {RAW_DATA_FILE_PATH}")

print(f"Gefundene Probenblätter: {probe_sheets}")

# Sammeln der Ergebnisse
gesamt_ergebnisse = []

# Durchlaufen der gefilterten Blätter
for sheet_name in probe_sheets:
    print(f"\n=== Verarbeite Blatt: {sheet_name} ===")

    # Daten vom aktuellen Blatt laden (Rohdaten beginnen ab Zeile 3)
    data = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
    data.columns = ['mm', 'N']  # Umbenennen der Spalten

    # Deflexion (mm) und Kraft (N)
    x = data['mm']
    y = data['N']

    # Ermittlung des Punktes mit der größten Kraft
    max_force_index = y.idxmax()
    max_force = y[max_force_index]
    one_third_max_force = max_force / 3
    one_tenth_max_force = max_force / 10

    # Finden des x-Werts für 1/3 der maximalen Kraft
    closest_index = (y - one_third_max_force).abs().idxmin()
    x_value_at_one_third_max_force = x[closest_index]

    # Finden des x-Werts für 1/10 der maximalen Kraft
    closest_index = (y - one_tenth_max_force).abs().idxmin()
    x_value_at_one_tenth_max_force = x[closest_index]

    print(f"Größte Kraft = {max_force:.2f} N bei x={x[max_force_index]:.3f} mm")
    print(f"Ein Drittel der maximalen Kraft: {one_third_max_force:.2f} N")
    print(f"Dazugehöriger x-Wert: {x_value_at_one_third_max_force:.3f} mm")
    print(f"Ein Zehntel der maximalen Kraft: {one_tenth_max_force:.2f} N")
    print(f"Dazugehöriger x-Wert: {x_value_at_one_tenth_max_force:.3f} mm")

    # Filtern der Daten für die Analyse (bis zum Maximum)
    x_filtered = x[:max_force_index + 1].values
    y_filtered = y[:max_force_index + 1].values

    # Filtern der unsauberen Anfangsdaten: Alle Werte mit x < 0,0005 mm werden entfernt
    min_deflection = 0.0005
    mask = x_filtered >= min_deflection
    x_filtered = x_filtered[mask]
    y_filtered = y_filtered[mask]

    # Polynom-Glättung 4. Grades
    degree = 4
    p_coeff = np.polyfit(x_filtered, y_filtered, degree)
    y_smooth_poly = np.polyval(p_coeff, x_filtered)

    # Berechnung des r²-Werts für die Polynom-Glättung
    SS_res = np.sum((y_filtered - y_smooth_poly) ** 2)
    SS_tot = np.sum((y_filtered - np.mean(y_filtered)) ** 2)
    r2_poly = 1 - SS_res / SS_tot
    print(f"r² der Polynom-Glättung (Grad {degree}): {r2_poly:.4f}")

    # Lineare Regression im spezifizierten Bereich
    regression_start_point_mm = x_value_at_one_tenth_max_force
    regression_end_point_mm = x_value_at_one_third_max_force

    regression_mask = (x_filtered >= regression_start_point_mm) & (x_filtered <= regression_end_point_mm)
    x_regression = x_filtered[regression_mask].reshape(-1, 1)
    y_regression = y_filtered[regression_mask]

    linear_model = LinearRegression()
    linear_model.fit(x_regression, y_regression)

    slope = linear_model.coef_[0]
    intercept = linear_model.intercept_
    print(f"Funktionsgleichung der Geraden: y = {slope:.4f} * x + {intercept:.4f}")

    x_trendline = np.linspace(0, x.max(), 200).reshape(-1, 1)
    y_trend = linear_model.predict(x_trendline)
    r_squared = linear_model.score(x_regression, y_regression)

    # Y-Werte der Punkte der Regressionsgrenzen
    regression_start_point = (float(regression_start_point_mm), float(slope * regression_start_point_mm + intercept))
    regression_end_point = (float(regression_end_point_mm), float(slope * regression_end_point_mm + intercept))

    # Berechnung des Elastizitätsmoduls (E-Modul) aus der Regression
    L = 300  # Beispielwert in mm
    b = 20   # Beispielwert in mm
    h = 20   # Beispielwert in mm
    if L > 0 and b > 0 and h > 0:
        computed_E_modul = (L ** 3 * slope) / (4 * b * h ** 3)
    else:
        computed_E_modul = None

    if computed_E_modul is not None:
        computed_E_modul_GPa = computed_E_modul / 1000.0
        print(f"E-Modul für {sheet_name}: {computed_E_modul_GPa:.2f} GPa")
    else:
        print(f"E-Modul für {sheet_name} konnte nicht berechnet werden.")

    # Abweichungsanalyse (rückwärts)
    x_backward = x_filtered.copy()
    y_smooth_backward = y_smooth_poly.copy()

    sort_idx = np.argsort(x_backward)[::-1]  # absteigend sortiert
    x_backward = x_backward[sort_idx]
    y_smooth_backward = y_smooth_backward[sort_idx]

    # Lineare Regression rückwärts anwenden
    y_pred_backward = linear_model.predict(x_backward.reshape(-1, 1))

    # Script Parameter einstellen
    CLOSE_THRESHOLD = 0.01  # 0.01 = 1% | Stelle, wo sich Polynom-Glättung und Regression annähern
    deviations_bw = np.abs(y_smooth_backward - y_pred_backward)
    deviation_ratios_bw = deviations_bw / y_smooth_backward
    mask_bw_close = (deviation_ratios_bw < CLOSE_THRESHOLD)
    close_indices_bw = np.where(mask_bw_close)[0]

    if len(close_indices_bw) > 0:
        first_close_index = close_indices_bw[0]
        close_x = x_backward[first_close_index]
        close_smooth_y = y_smooth_backward[first_close_index]
        print(f"Rückwärts: Ab x={close_x:.4f} mm ist Abweichung < {CLOSE_THRESHOLD * 100:.1f}% (Glättung={close_smooth_y:.4f} N)")
    else:
        close_x = None
        first_close_index = 0  # Fallback für Plot
        close_smooth_y = None
        print("Keine Annäherung (rückwärts) < close_threshold gefunden.")

    # Berechnung der Flächen und Sprödigkeitsindex
    if close_x is not None:
        first_close_index = np.where(x_filtered >= close_x)[0][0]
        area_1 = trapezoid(y_smooth_poly[:first_close_index + 1], x_filtered[:first_close_index + 1])
        area_2 = trapezoid(y_smooth_poly[first_close_index:], x_filtered[first_close_index:])
    else:
        area_1 = 0
        area_2 = trapezoid(y_smooth_poly, x_filtered)

    total_area = area_1 + area_2
    sprödigkeitsindex = (area_1 / total_area) * 100 if total_area > 0 else 0
    print(f"Fläche 1 = {area_1:.4f}, Fläche 2 = {area_2:.4f}, Gesamtfläche = {total_area:.4f}, Sprödigkeitsindex = {sprödigkeitsindex:.2f}%")

    # Ergebnisse für das aktuelle Blatt speichern
    gesamt_ergebnisse.append({
        "Probe": sheet_name,
        "F(max) [N]": max_force,
        "dL bei F(max) [mm]": x[max_force_index],
        "m": slope,
        "F(high) [N] = F/3": regression_end_point,
        "F(low) [N] = F/10": regression_start_point,
        "dL bei F(high) [mm]": regression_end_point_mm,
        "dL bei F(low) [mm]": regression_start_point_mm,
        "R²(gerade)": r_squared,
        "R²(poly)": r2_poly,
        "S(annäherung) [%]": CLOSE_THRESHOLD,
        "σ(prop) [mm]": close_x,
        "W(elastisch) [N*mm]": area_1,
        "W(plastisch) [N*mm]": area_2,
        "W(gesamt) [N*mm]": total_area,
        "I(sprödigkeit) [%]": sprödigkeitsindex,
        "E-Modul [GPa]": computed_E_modul_GPa
    })

    # Visualisierung der Analyseergebnisse
    plt.figure(figsize=(12, 6))

    # Bereiche visuell hervorheben
    plt.fill_between(
        x_filtered[:first_close_index + 1], 0, y_smooth_poly[:first_close_index + 1],
        alpha=0.1, color='red', label='W(elastisch)'
    )
    plt.fill_between(
        x_filtered[first_close_index:], 0, y_smooth_poly[first_close_index:],
        alpha=0.1, color='green', label='W(plastisch)'
    )

    # Hauptdatenplot
    plt.plot(x, y, label='Rohdaten', color='black', alpha=0.7)
    plt.plot(x_filtered, y_smooth_poly, label='Polynomfit', color='blue')
    plt.axvline(x=x[max_force_index], color='black', linestyle='--')
    plt.plot(x[max_force_index], y[max_force_index], 'ko', label='F(max)')
    plt.plot(x_trendline, y_trend, color='red', linestyle='--', label=f'Lineare Regression')

    # Markierungen für Regressionsgrenzen
    plt.plot(regression_start_point[0], regression_start_point[1], 'ro', label='F(low)')
    plt.plot(regression_end_point[0], regression_end_point[1], 'ro', label='F(high)')

    # Markierung der Annäherung (rückwärts)
    if close_x is not None:
        plt.axvline(x=close_x, color='teal', linestyle='--')
        plt.plot(close_x, close_smooth_y, 'go', label=f'Proportionalitätsgrenze')

    # Achsenbeschriftungen, Titel, Gitter und Legende
    plt.xlabel('Verformung [mm]')
    plt.ylabel('Standardkraft [N]')
    plt.title(f"Analyse für {sheet_name}")
    plt.legend(loc='best')
    plt.grid(True)

    # Annotieren von weiteren Informationen im Plot
    plt.text(
        0.5, 0.95, f"Sprödigkeitsindex = {sprödigkeitsindex:.2f}%",
        fontsize=12, color='blue', ha='center', va='top', transform=plt.gca().transAxes
    )
    plt.text(
        0.5, 0.90, f"Annäherungsschwellenwert = {CLOSE_THRESHOLD * 100:.1f}%",
        fontsize=10, color='black', ha='center', va='top', transform=plt.gca().transAxes
    )
    plt.text(
        0.5, 0.85, f"F(low) = F(max)/10",
        fontsize=10, color='black', ha='center', va='top', transform=plt.gca().transAxes
    )
    plt.text(
        0.5, 0.80, f"F(high) = F(max)/3",
        fontsize=10, color='black', ha='center', va='top', transform=plt.gca().transAxes
    )

    # Jeden Plot speichern
    full_output_path = os.path.join(OUTPUT_PLOT_DIRECTORY, f"{sheet_name}.png")
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')


# Ergebnisse in Excel speichern
ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
ergebnisse_df.to_excel(OUTPUT_FILE_EXCEL, index=False)
print(f"\nAnalyse abgeschlossen. Ergebnisse in '{OUTPUT_FILE_EXCEL}' gespeichert.")

# Plots zeigen
if SHOW_PLOTS:
    plt.show()