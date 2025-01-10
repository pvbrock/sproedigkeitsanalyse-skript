import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy import trapezoid
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import io

def load_probe_sheets(file_path):
    """Load probe sheets from the Excel file."""
    xls = pd.ExcelFile(file_path)
    probe_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith("Probe")]
    print(f"Gefundene Probenblätter: {probe_sheets}")
    return xls, probe_sheets

def process_probe_sheet(xls, sheet_name):
    """Process a single probe sheet and return the results and processed data."""
    data = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
    data.columns = ['mm', 'N']

    x = data['mm']
    y = data['N']

    # Determine break point
    max_force_index = y.idxmax()
    bruch_threshold = 100 * y[max_force_index]
    bruch_indices = np.where(y[max_force_index:] < bruch_threshold)[0]
    bruch_index = max_force_index + bruch_indices[0] if len(bruch_indices) > 0 else len(y) - 1

    x_filtered = x[:bruch_index].values
    y_filtered = y[:bruch_index].values

    # Ensure monotonic increase in x_filtered
    x_filtered, y_filtered = zip(*[(xi, yi) for xi, yi in zip(x_filtered, y_filtered) if xi > x_filtered[0]])
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

    # Smooth data with Univariate Spline
    spline = UnivariateSpline(x_filtered, y_filtered)
    spline.set_smoothing_factor(10000000)
    y_smooth_spline = spline(x_filtered)

    # Calculate slope and proportional limit
    spline_derivative = spline.derivative()
    spline_slopes = spline_derivative(x_filtered)

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

    # Calculate areas
    if prop_limit_index is not None:
        area_1 = trapezoid(y_smooth_spline[:prop_limit_index + 1], x_filtered[:prop_limit_index + 1])
        area_2 = trapezoid(y_smooth_spline[prop_limit_index:], x_filtered[prop_limit_index:])
    else:
        area_1 = 0
        area_2 = trapezoid(y_smooth_spline, x_filtered)

    total_area = area_1 + area_2
    sprödigkeitsindex = (area_1 / total_area) * 100 if total_area > 0 else 0

    results = {
        "Maximale Kraft (mm)": x[bruch_index],
        "Proportionalitätsgrenze (mm)": prop_limit_mm,
        "Fläche 1 (N·mm)": area_1,
        "Fläche 2 (N·mm)": area_2,
        "Sprödigkeitsindex (%)": sprödigkeitsindex
    }

    processed_data = {
        'mm': x_filtered,
        'N': y_filtered,
        'Geglättete Kraft-Durchbiegungskurve': y_smooth_spline
    }

    return results, processed_data, (x, y, x_filtered, y_smooth_spline, bruch_index, prop_limit_mm, prop_limit_index)

def save_results_to_excel(output_file, probe_sheets, all_results, processed_data_list):
    """Save the analysis results and processed data to an Excel file."""
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, data, results in zip(probe_sheets, processed_data_list, all_results):
            results_df = pd.DataFrame(data)
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Summary results
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(writer, sheet_name="Ergebnisse", index=False)

def add_plots_to_excel(output_file, probe_sheets, plot_data_list):
    """Add plots to the Excel file."""
    wb = load_workbook(output_file)
    for sheet_name, plot_data in zip(probe_sheets, plot_data_list):
        x, y, x_filtered, y_smooth_spline, bruch_index, prop_limit_mm, prop_limit_index = plot_data
        ws = wb[sheet_name]

        # Create plot
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='Rohdaten', color='gray', alpha=0.5)
        plt.plot(x_filtered, y_smooth_spline, label='Geglättete Kraft-Durchbiegungskurve', color='blue')
        plt.axvline(x=x[bruch_index], color='green', linestyle='--', label='Maximale Kraft (mm)')

        if prop_limit_mm is not None:
            plt.axvline(x=prop_limit_mm, color='red', linestyle='--', label='Proportionalitätsgrenze (mm)')
            plt.fill_between(x_filtered[:prop_limit_index + 1], 0, y_smooth_spline[:prop_limit_index + 1], color='green', alpha=0.2, label='Fläche 1 (N·mm)')
            plt.fill_between(x_filtered[prop_limit_index:], 0, y_smooth_spline[prop_limit_index:], color='orange', alpha=0.2, label='Fläche 2 (N·mm)')

        plt.xlabel('Deflection (mm)')
        plt.ylabel('Force (N)')
        plt.title(f"Analyse für {sheet_name}")
        plt.legend()
        plt.grid(True)

        # Save plot to byte stream
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        plt.close()

        # Add image to Excel
        img = Image(img_stream)
        img.anchor = 'H1'
        ws.add_image(img)

    wb.save(output_file)

def main():
    input_file = "zwick_data_all.xlsx" #TODO: adapt to read .xls files
    #input_file = "sproedigkeit_direkt_von_zwick_zwei_proben.xlsx"
    output_file = "Analyse_Ergebnisse.xlsx"

    xls, probe_sheets = load_probe_sheets(input_file)

    all_results = []
    processed_data_list = []
    plot_data_list = []

    for sheet_name in probe_sheets:
        print(f"Verarbeite Blatt: {sheet_name}")
        results, processed_data, plot_data = process_probe_sheet(xls, sheet_name)
        all_results.append({**results, "Probe": sheet_name})
        processed_data_list.append(processed_data)
        plot_data_list.append(plot_data)

    save_results_to_excel(output_file, probe_sheets, all_results, processed_data_list)
    print("Speichern von Diagrammen jeder Probe")
    add_plots_to_excel(output_file, probe_sheets, plot_data_list)
    print("Analyse abgeschlossen. Ergebnisse und Diagramme in 'Analyse_Ergebnisse.xlsx' gespeichert.")

if __name__ == "__main__":
    main()
