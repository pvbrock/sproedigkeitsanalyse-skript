import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapezoid
from sklearn.linear_model import LinearRegression

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
    # Simulate a single "sheet" for CSV files
    csv_data = pd.read_csv(data_file)
    probe_sheets = ["CSV_Data"]
else:
    raise ValueError(f"Unsupported file format: {data_file}")

#print(f"Gefundene Probenblätter: {probe_sheets}")

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
    #print(f"Größte Kraft = {max_force:.2f} N bei x={x[max_force_index]:.3f} mm")

    # Daten für Analyse filtern
    x_filtered = x[:max_force_index + 1].values
    y_filtered = y[:max_force_index + 1].values

    # Auf streng steigende x filtern (manche Messungen laufen rückwärts oder enthalten Dubletten)
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
    # Spline-Werte im gesamten x_filtered-Bereich
    y_smooth_spline = spline(x_filtered)

    # -------------------------------------------------------------
    # 3) Lineare Regression im Bereich 0.5 mm bis 3.5 mm
    # -------------------------------------------------------------
    regression_start_point_mm = 0.5
    regression_end_point_mm = 3

    regression_mask = (x_filtered >= regression_start_point_mm) & (x_filtered <= regression_end_point_mm)
    x_regression = x_filtered[regression_mask].reshape(-1, 1)
    y_regression = y_filtered[regression_mask]

    linear_model = LinearRegression()
    linear_model.fit(x_regression, y_regression)

    slope = linear_model.coef_[0]
    intercept = linear_model.intercept_
    #print(f"Funktionsgleichung der Geraden: y = {slope:.4f} * x + {intercept:.4f}")

    x_trendline = np.linspace(0, x.max(), 200).reshape(-1, 1)
    y_trend = linear_model.predict(x_trendline)
    r_squared = linear_model.score(x_regression, y_regression)

    # Punkte der Regressionsgrenzen
    regression_start_point = (regression_start_point_mm, slope * regression_start_point_mm + intercept)
    regression_end_point = (regression_end_point_mm, slope * regression_end_point_mm + intercept)

    # -------------------------------------------------------------
    # 5) Abweichungsanalyse (rückwärts)
    #    Daten rückwärts sortieren, Abweichung > threshold
    # -------------------------------------------------------------
    x_backward = x_filtered.copy()
    y_smooth_backward = y_smooth_spline.copy()

    sort_idx = np.argsort(x_backward)[::-1]  # absteigend sortiert
    x_backward = x_backward[sort_idx]
    y_smooth_backward = y_smooth_backward[sort_idx]

    # Lineare Regression rückwärts anwenden
    y_pred_backward = linear_model.predict(x_backward.reshape(-1, 1))



    # -------------------------------------------------------------
    # Stelle, wo sich Spline und Regression annähern
    #    => Abweichung < close_threshold
    # -------------------------------------------------------------
    close_threshold = 0.01  # 2%
    deviations_bw = np.abs(y_smooth_backward - y_pred_backward)
    deviation_ratios_bw = deviations_bw / y_smooth_backward
    mask_bw_close = (deviation_ratios_bw < close_threshold)
    close_indices_bw = np.where(mask_bw_close)[0]

    if len(close_indices_bw) > 0:
        first_close_index = close_indices_bw[0]
        close_x = x_backward[first_close_index]
        close_spline_y = y_smooth_backward[first_close_index]
        #print(
            #f"Rückwärts: Ab x={close_x:.4f} mm ist Abweichung < {close_threshold * 100:.1f}% (Spline={close_spline_y:.4f} N)")
    else:
        close_x = None
        close_spline_y = None
        #print("Keine Annäherung (rückwärts) < close_threshold gefunden.")

    # -------------------------------------------------------------
    # 6) Berechnung der Flächen und Sprödigkeitsindex
    # -------------------------------------------------------------
    if close_x is not None:
        first_close_index = np.where(x_filtered >= close_x)[0][0]
        area_1 = trapezoid(y_smooth_spline[:first_close_index + 1], x_filtered[:first_close_index + 1])
        area_2 = trapezoid(y_smooth_spline[first_close_index:], x_filtered[first_close_index:])
    else:
        area_1 = 0
        area_2 = trapezoid(y_smooth_spline, x_filtered)

    total_area = area_1 + area_2
    sprödigkeitsindex = (area_1 / total_area) * 100 if total_area > 0 else 0
    #print(f"Fläche 1 = {area_1:.4f}, Fläche 2 = {area_2:.4f}, Sprödigkeitsindex = {sprödigkeitsindex:.2f}%")

    # Append results for the current probe
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
        "smooth Factor": spline_smooth_factor,
        "Fläche 1 (N*mm)": area_1,
        "Fläche 2 (N*mm)": area_2,
        "Sprödigkeitsindex": sprödigkeitsindex
    })

    # Visualization with legend for each graphic
    plt.figure(figsize=(12, 6))

    # Fill areas
    plt.fill_between(
        x_filtered[:first_close_index + 1], 0, y_smooth_spline[:first_close_index + 1],
        alpha=0.1, color='red', label='Fläche 1'
    )
    plt.fill_between(
        x_filtered[first_close_index:], 0, y_smooth_spline[first_close_index:],
        alpha=0.1, color='green', label='Fläche 2'
    )

    # Plot the main visualization
    plt.plot(x, y, label='Rohdaten (komplett)', color='black', alpha=0.7)
    plt.plot(x_filtered, y_smooth_spline, label='Spline Glättung', color='blue')
    plt.axvline(x=x[max_force_index], color='pink', linestyle='--', label='Max. Kraft')
    plt.plot(x_trendline, y_trend, color='red', linestyle='--',
             label=f'Lineare Regression (R²={r_squared:.4f})')
    plt.plot(x[max_force_index], y[max_force_index], 'ko', label='Max. Kraft Punkt')

    # Regression boundary points
    plt.plot(regression_start_point[0], regression_start_point[1], 'ro', label='Regression Startpunkt')
    plt.plot(regression_end_point[0], regression_end_point[1], 'ro', label='Regression Endpunkt')

    # Highlight point of backward approximation
    if close_x is not None:
        plt.axvline(x=close_x, color='teal', linestyle='--',
                    label=f'Annäherung < {close_threshold * 100:.1f}%')
        plt.plot(close_x, close_spline_y, 'go', label=f'Close (rückwärts) Spline-Pkt')

    # Add labels, title, grid, and legend
    plt.xlabel('Deflection (mm)')
    plt.ylabel('Force (N)')
    plt.title(f"Analyse für {sheet_name}")
    plt.legend(loc='best')  # Add legend here
    plt.grid(True)

    # Annotate Sprödigkeitsindex on the plot
    plt.text(
        0.5, 0.95, f"Sprödigkeitsindex: {sprödigkeitsindex:.2f}%",
        fontsize=12, color='darkred', ha='center', va='top', transform=plt.gca().transAxes
    )



# -------------------------------------------------------------
# 8) Ergebnisse in Excel speichern
# -------------------------------------------------------------



import pandas as pd
from openpyxl import load_workbook

# Assuming 'ergebnisse_df' contains the data you want to add
new_sheet_name = f"smooth({spline_smooth_factor:.1e})_schwelle({close_threshold})"  # Define the name of the new sheet
existing_file = "Analyse_Ergebnisse_Lineare_Regression_alle_Chargen.xlsx"  # The existing Excel file

# Load the existing workbook and write to a new sheet
ergebnisse_df = pd.DataFrame(gesamt_ergebnisse)
try:
    with pd.ExcelWriter(existing_file, engine="openpyxl", mode="a") as writer:
        ergebnisse_df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    print(f"Data successfully written to the sheet '{new_sheet_name}' in '{existing_file}'.")
except FileNotFoundError:
    print(f"File '{existing_file}' does not exist. Creating a new one.")
    ergebnisse_df.to_excel(existing_file, sheet_name=new_sheet_name, index=False)


#plt.show()