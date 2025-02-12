import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from sklearn.linear_model import LinearRegression

# Path to data file (can be .xlsx, .xls, or .csv)
data_file = "paul-master-manuell-geaendert-Xcf052_BU_CA2.xls"  # Change to your file path

# Check file type and load accordingly
if data_file.endswith(".xlsx"):
    xls = pd.ExcelFile(data_file)
    probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
elif data_file.endswith(".xls"):
    xls = pd.ExcelFile(data_file, engine="xlrd")
    probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
elif data_file.endswith(".csv"):
    # Simulate a single "sheet" for CSV files
    csv_data = pd.read_csv(data_file)
    probe_sheets = ["CSV_Data"]
else:
    raise ValueError(f"Unsupported file format: {data_file}")

print(f"Gefundene Probenblätter: {probe_sheets}")

# Excel-Datei laden und Blätter filtern
xls = pd.ExcelFile(data_file)
probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
print(f"Gefundene Probenblätter: {probe_sheets}")

# Ergebnisse sammeln
gesamt_ergebnisse = []

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
    max_force_index = y.idxmax()  # Index der größten Kraft
    max_force = y[max_force_index]
    print(f"Größte Kraft = {max_force:.2f} N bei x={x[max_force_index]:.3f} mm")

    # Daten für Analyse filtern (bis zum Maximum)
    x_filtered = x[:max_force_index + 1].values
    y_filtered = y[:max_force_index + 1].values

    # -------------------------------------------------------------
    # Filtere unsaubere Anfangsdaten: Alle Werte mit x < 0,0005 mm werden entfernt
    # -------------------------------------------------------------
    min_deflection = 0.0005  # in mm
    mask = x_filtered >= min_deflection
    x_filtered = x_filtered[mask]
    y_filtered = y_filtered[mask]

    # -------------------------------------------------------------
    # 2) Polynom-Glättung 4. Grades (statt Spline-Glättung)
    # -------------------------------------------------------------
    degree = 4
    # Fit des Polynoms 4. Grades an die gefilterten Daten
    p_coeff = np.polyfit(x_filtered, y_filtered, degree)
    # Berechnung der geglätteten Werte über den gesamten x_filtered-Bereich
    y_smooth_poly = np.polyval(p_coeff, x_filtered)

    # Berechnung des MSE für die Polynom-Glättung
    mse_poly = np.mean((y_filtered - y_smooth_poly) ** 2)
    print(f"MSE der Polynom-Glättung (Grad {degree}): {mse_poly:.4f}")

    # Berechnung des r²-Werts für die Polynom-Glättung
    SS_res = np.sum((y_filtered - y_smooth_poly) ** 2)
    SS_tot = np.sum((y_filtered - np.mean(y_filtered)) ** 2)
    r2_poly = 1 - SS_res / SS_tot
    print(f"r² der Polynom-Glättung (Grad {degree}): {r2_poly:.4f}")

    # -------------------------------------------------------------
    # 3) Lineare Regression im Bereich 0.5 mm bis 3 mm
    # -------------------------------------------------------------
    regression_start_point_mm = 1
    regression_end_point_mm = 3

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

    # Punkte der Regressionsgrenzen
    regression_start_point = (regression_start_point_mm, slope * regression_start_point_mm + intercept)
    regression_end_point = (regression_end_point_mm, slope * regression_end_point_mm + intercept)

    # -------------------------------------------------------------
    # 4) Berechnung des Elastizitätsmoduls (E-Modul) aus der Regression
    #    Stützweite (L), Breite (b), Höhe (h) -> Falls bekannt, hier Werte eintragen!
    # -------------------------------------------------------------
    L = 300  # Beispielwert in mm
    b = 20   # Beispielwert in mm
    h = 20   # Beispielwert in mm
    if L > 0 and b > 0 and h > 0:
        # Berechnung in N/mm² (gleich MPa)
        computed_E_modul = (L ** 3 * slope) / (4 * b * h ** 3)
    else:
        computed_E_modul = None

    # Umrechnung in GPa (1 GPa = 1000 N/mm²)
    if computed_E_modul is not None:
        computed_E_modul_GPa = computed_E_modul / 1000.0
        print(f"E-Modul für {sheet_name}: {computed_E_modul_GPa:.2f} GPa")
    else:
        computed_E_modul_GPa = None
        print(f"E-Modul für {sheet_name} konnte nicht berechnet werden.")

    # -------------------------------------------------------------
    # 5) Abweichungsanalyse (rückwärts)
    #    Daten rückwärts sortieren, Abweichung > threshold
    # -------------------------------------------------------------
    x_backward = x_filtered.copy()
    y_smooth_backward = y_smooth_poly.copy()

    sort_idx = np.argsort(x_backward)[::-1]  # absteigend sortiert
    x_backward = x_backward[sort_idx]
    y_smooth_backward = y_smooth_backward[sort_idx]

    # Lineare Regression rückwärts anwenden
    y_pred_backward = linear_model.predict(x_backward.reshape(-1, 1))

    # -------------------------------------------------------------
    # Stelle, wo sich Polynom-Glättung und Regression annähern
    #    => Abweichung < close_threshold
    # -------------------------------------------------------------
    close_threshold = 0.005  # 2%
    deviations_bw = np.abs(y_smooth_backward - y_pred_backward)
    deviation_ratios_bw = deviations_bw / y_smooth_backward
    mask_bw_close = (deviation_ratios_bw < close_threshold)
    close_indices_bw = np.where(mask_bw_close)[0]

    if len(close_indices_bw) > 0:
        first_close_index = close_indices_bw[0]
        close_x = x_backward[first_close_index]
        close_smooth_y = y_smooth_backward[first_close_index]
        print(f"Rückwärts: Ab x={close_x:.4f} mm ist Abweichung < {close_threshold * 100:.1f}% (Glättung={close_smooth_y:.4f} N)")
    else:
        close_x = None
        first_close_index = 0  # Fallback für Plot
        close_smooth_y = None
        print("Keine Annäherung (rückwärts) < close_threshold gefunden.")

    # -------------------------------------------------------------
    # 6) Berechnung der Flächen und Sprödigkeitsindex
    # -------------------------------------------------------------
    if close_x is not None:
        first_close_index = np.where(x_filtered >= close_x)[0][0]
        area_1 = trapezoid(y_smooth_poly[:first_close_index + 1], x_filtered[:first_close_index + 1])
        area_2 = trapezoid(y_smooth_poly[first_close_index:], x_filtered[first_close_index:])
    else:
        area_1 = 0
        area_2 = trapezoid(y_smooth_poly, x_filtered)

    total_area = area_1 + area_2
    sprödigkeitsindex = (area_1 / total_area) * 100 if total_area > 0 else 0
    print(f"Fläche 1 = {area_1:.4f}, Fläche 2 = {area_2:.4f}, Sprödigkeitsindex = {sprödigkeitsindex:.2f}%")

    # Ergebnisse für das aktuelle Blatt speichern
    gesamt_ergebnisse.append({
        "Probe": sheet_name,
        "Max. Kraft (N)": max_force,
        "Max. Kraft (mm)": x[max_force_index],
        "Regression Steigung": slope,
        "Regression Achsenabschnitt": intercept,
        "Regression Startpunkt mm": regression_start_point_mm,
        "Regression Endpunkt mm": regression_end_point_mm,
        "Regression Bestimmtheitsmaß (R²)": r_squared,
        "Annäherung (rückwärts, Schwellenwert)": close_threshold,
        "Annäherung (rückwärts, mm)": close_x,
        "Fläche 1 (N*mm)": area_1,
        "Fläche 2 (N*mm)": area_2,
        "Sprödigkeitsindex": sprödigkeitsindex,
        "Poly MSE": mse_poly,
        "Poly r²": r2_poly,
        "E-Modul (GPa)": computed_E_modul_GPa
    })

    # Visualization
    plt.figure(figsize=(12, 6))

    # Bereiche füllen
    plt.fill_between(
        x_filtered[:first_close_index + 1], 0, y_smooth_poly[:first_close_index + 1],
        alpha=0.1, color='red', label='Fläche 1'
    )
    plt.fill_between(
        x_filtered[first_close_index:], 0, y_smooth_poly[first_close_index:],
        alpha=0.1, color='green', label='Fläche 2'
    )

    # Hauptplot
    plt.plot(x, y, label='Rohdaten (komplett)', color='black', alpha=0.7)
    plt.plot(x_filtered, y_smooth_poly, label='Polynom-Glättung (Grad 4)', color='blue')
    plt.axvline(x=x[max_force_index], color='pink', linestyle='--', label='Max. Kraft')
    plt.plot(x_trendline, y_trend, color='red', linestyle='--',
             label=f'Lineare Regression (R²={r_squared:.4f})')
    plt.plot(x[max_force_index], y[max_force_index], 'ko', label='Max. Kraft Punkt')

    # Regressionsgrenzen
    plt.plot(regression_start_point[0], regression_start_point[1], 'ro', label='Regression Startpunkt')
    plt.plot(regression_end_point[0], regression_end_point[1], 'ro', label='Regression Endpunkt')

    # Markierung der Annäherung (rückwärts)
    if close_x is not None:
        plt.axvline(x=close_x, color='teal', linestyle='--',
                    label=f'Annäherung < {close_threshold * 100:.1f}%')
        plt.plot(close_x, close_smooth_y, 'go', label=f'Close (rückwärts) Glättung-Pkt')

    # Achsenbeschriftung, Titel, Gitter und Legende
    plt.xlabel('Deflection (mm)')
    plt.ylabel('Force (N)')
    plt.title(f"Analyse für {sheet_name}")
    plt.legend(loc='best')
    plt.grid(True)

    # Annotiere Sprödigkeitsindex, Poly MSE, Poly r² und E-Modul (in GPa) im Plot
    plt.text(
        0.5, 0.95, f"Sprödigkeitsindex: {sprödigkeitsindex:.2f}%",
        fontsize=12, color='darkred', ha='center', va='top', transform=plt.gca().transAxes
    )
    plt.text(
        0.5, 0.90, f"MSE (Poly): {mse_poly:.4f}",
        fontsize=12, color='blue', ha='center', va='top', transform=plt.gca().transAxes
    )
    plt.text(
        0.5, 0.85, f"r² (Poly): {r2_poly:.4f}",
        fontsize=12, color='blue', ha='center', va='top', transform=plt.gca().transAxes
    )
    if computed_E_modul_GPa is not None:
        plt.text(
            0.5, 0.80, f"E-Modul: {computed_E_modul_GPa:.2f} GPa",
            fontsize=12, color='purple', ha='center', va='top', transform=plt.gca().transAxes
        )

    plt.show()

# -------------------------------------------------------------
# Ergebnisse in Excel speichern
# -------------------------------------------------------------
ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
output_file = f"Analyse_Ergebnisse_Lineare_Regression_BU_CA_poly(degree{degree})_schwelle({close_threshold})re({regression_start_point_mm}-{regression_end_point_mm}).xlsx"
ergebnisse_df.to_excel(output_file, index=False)
print(f"\nAnalyse abgeschlossen. Ergebnisse in '{output_file}' gespeichert.")
