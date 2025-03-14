# 📦 Installation & Setup

Dieses Projekt benötigt ein Conda-Environment mit Python 3.12 und allen Abhängigkeiten aus der `requirements.txt`.

## Schritte zur Installation

### 1. Repository klonen (falls relevant)

```bash
git clone https://github.com/pvbrock/sproedigkeitsanalyse-skript.git
cd sproedigkeitsanalyse-skript

```

---

### 2. Conda-Environment erstellen

Erstelle ein neues Conda-Environment mit dem Namen **sproedigkeit** und Python 3.12:

```bash
conda create -n sproedigkeit python=3.12
```

---

### 3. Conda-Environment aktivieren

```bash
conda activate sproedigkeit
```

---

### 4. Anforderungen installieren (requirements.txt)

```bash
pip install -r requirements.txt
```

# 🛠️ Konfiguration von Parametern

Nach Durchführung von Installation & Setup oben

### 1. Pfad für Rohdaten festlegen in Variable `RAW_DATA_FILE_PATH`

```
RAW_DATA_FILE_PATH = r"ordner\unterordner\rohdaten_beispiel.xls"
```
### 2. Annäherungsschwellenwert festlegen in in Variable `CLOSE_THRESHOLD`

```
CLOSE_THRESHOLD = 0.01
```
### 3. Startpunkt der linearen Regression festlegen in Variable `regression_start_point_mm`
Der Startpunkt wird aktuell definiert durch Variable`x_value_at_one_tenth_max_force` = x-Wert bei einem Zehntel der maximalen Kraft der Probe in millimeter
```
regression_start_point_mm = x_value_at_one_tenth_max_force
    
```
### 4. Endpunkt der linearen Regression festlegen in Variable `regression_end_point_mm`
Der Endpunkt wird aktuell definiert durch Variable `x_value_at_one_third_max_force` = x-Wert bei einem drittel der maximalen Kraft der Probe in millimeter
```
regression_end_point_mm = x_value_at_one_third_max_force
```
### 5. Pfad für Analyseergebnisse festlegen in Variable `OUTPUT_FILE_EXCEL`

Kann als Dateiname angegeben werden oder als Pfad in anderem Ordner, siehe Beispiele:

```
OUTPUT_FILE_EXCEL = r"analyse_ergebnisse.xlsx"
OUTPUT_FILE_EXCEL = r"ordner\unterordner\analyse_ergebnisse.xlsx"
OUTPUT_FILE_EXCEL = r"C:\_prog\_code\sproedigkeitsanalyse-skript\output\analyse_ergebnisse.xlsx"
```


### 6. Plots anzeigen lassen oder nicht mit `SHOW_PLOTS`

Zeige Plots an mit: 
```
SHOW_PLOTS = True
```

Zeige Plots nicht an mit: 
```
SHOW_PLOTS = False
```

### 7. Pfad für Speicherung der Plots festlegen in Variable `OUTPUT_PLOT_DIRECTORY`

```
OUTPUT_PLOT_DIRECTORY = r"output\plots"
OUTPUT_PLOT_DIRECTORY = r"C:\_prog\_code\ordner\plots"
```

# 🚀 Ausführung vom Script

### 1. Script ausführen

```
python sproedigkeitsanalyse-script.py
``` 