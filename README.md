# 📦 Installation & Setup

Dieses Projekt benötigt ein Conda-Environment mit Python 3.12 und allen Abhängigkeiten aus der `requirements.txt`.

## 🔧 Schritte zur Installation

### 1. Repository klonen (falls relevant)

```bash
git clone https://github.com/sdlaknow/sproedigkeit_edits.git
cd sproedigkeit_edits

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

# 🚀 Ausführung vom Script

Nach Durchführung von Installation & Setup oben

### 1. Pfad für Rohdaten festlegen in Variable `data_file`

```
data_file = r"ordner/unterordner/rohdaten.xls"
```
### 2. Annäherungsschwellenwert festlegen in in Variable `close_threshold`

```
close_thershold = 0,01
```
### 3. Startpunkt der linearen Regression festlegen in Variable `regression_start_point_mm`
Der Startpunkt wird definiert  durch Variable`x_value_at_one_tenth_max_force` = x-Wert bei einem Zehntel der maximalen Kraft
```
regression_start_point_mm = x_value_at_one_tenth_max_force
    
```
### 4. Endpunkt der linearen Regression festlegen in Variable `regression_end_point_mm`
Der Endpunkt wird definiert  durch Variable `x_value_at_one_third_max_force` = x-Wert bei einem drittel der maximalen Kraft
```
regression_end_point_mm = x_value_at_one_third_max_force
```
### 5. Pfad für Analyseergebnisse festlegen in Variable `output_file`

```
output_file = r"ordner/unterordner/analyse_ergebnisse.xls"
```

### 5. Script ausführen

```
python sproedigkeitsanalyse-script.py
``` 