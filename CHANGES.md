# CHANGES — CrashVision M1 (Adaptación local)

## 1. `requirements.txt` — creado

Generado a partir de las dependencias del notebook `retrain_yolov8_m1.ipynb`:

| Paquete | Versión mínima | Notas |
|---|---|---|
| `ultralytics` | 8.0.0 | YOLOv8 |
| `roboflow` | 1.1.0 | Descarga del dataset |
| `torch` | 2.11.0 | CUDA 12.8 — RTX 4060 Laptop |
| `torchvision` | 0.26.0 | CUDA 12.8 |
| `psutil` | 5.9.0 | Verificación de RAM |
| `numpy` | 1.24.0 | — |
| `pandas` | 2.0.0 | — |
| `matplotlib` | 3.7.0 | — |
| `pyyaml` | 6.0 | Lectura de `data.yaml` de Roboflow |

> Instalar PyTorch con GPU:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> ```

---

## 2. Entorno virtual `venv/` — creado

```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Python 3.11.9
- Torch reinstalado con CUDA 12.8 (`torch 2.11.0+cu128`) para la RTX 4060 Laptop
- CUDA verificado: `torch.cuda.is_available() → True`

---

## 3. `retrain_yolov8_m1.ipynb` — adaptado para ejecución local

### Sección 0 — Verificación del entorno
- Agregado `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` **antes de `import torch`** para evitar que VS Code/Jupyter inyecte la variable vacía y oculte la GPU.
- Eliminado `sys.exit()` cuando no se detecta GPU — ahora solo muestra advertencia y continúa.

### Sección 1 — Configuración de carpetas (antes: "Montaje de Google Drive")
- Eliminado `from google.colab import drive` y `drive.mount()`.
- Rutas absolutas locales reemplazando las rutas de Colab (`/content/drive/MyDrive/...`):
  ```
  BASE_DIR = C:\Users\dddma\OneDrive\Escritorio\Codigos_repors\x\CrashVision\
  ├── models\          ← baselines originales
  │   └── m1\          ← modelos M1 re-entrenados
  └── results\
      ├── yolov8n_m1\
      ├── yolov8s_m1\
      └── comparison_m1\
  ```

### Sección 2 — Dependencias
- Eliminado `!pip install ultralytics roboflow --quiet` (magic shell de Colab — incompatible con Jupyter local y ya instalado en el `venv`).

### Secciones 4 y 5 — Entrenamiento YOLOv8n y YOLOv8s
- `DEVICE` cambiado de `0` (hardcoded Colab) a detección automática `0 if torch.cuda.is_available() else 'cpu'`, luego fijado a `DEVICE = 0` al confirmar GPU disponible.
- Agregado `import torch` explícito en ambas celdas.

### Sección 8 — Exportación (antes: "Exportación a Google Drive")
- Rutas de destino cambiadas a carpetas locales usando `os.path.join()`.
- Eliminada la verificación de baselines originales al final (podía fallar si la carpeta no existía aún).

### Sección 9 — Resumen final
- Eliminadas referencias a "Google Drive" en los prints de salida.
- Rutas de archivos generados actualizadas a paths locales.

---

## 4. PyTorch — reinstalado con soporte CUDA

| | Antes | Después |
|---|---|---|
| Versión | `2.12.1` (CPU-only) | `2.11.0+cu128` |
| CUDA | No | Sí (CUDA 12.8) |
| GPU detectada | `False` | `True` — RTX 4060 Laptop |

```
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## 5. `train_m3.ipynb` — creado (M3: dataset representativo del dominio)

Clonado de `train_m2.ipynb` (misma estructura Secciones 0–9, rutas `*_M3`/carpeta `m3`, celda
idempotente `FORCE_RETRAIN`, export a `CrashVision/models/m3/`). Motivación: M1 (más epochs) y M2
(augmentation) **regresaron en video real** (6/8 → 4/8 → 2/8) → el cuello es **brecha de dominio**,
no el train. M3 ataca por los **datos**, no por hiperparámetros.

### Diferencias vs `train_m2.ipynb`

| | M2 | M3 |
|---|---|---|
| Datos | Roboflow v2 | **Roboflow v2 + fork dashcam** `car-accident-dashcam-j7efi` |
| Augmentation | dirigido (agresivo) | **default** (M2 fue el peor en video real) |
| Única palanca | augmentation | **el dataset** (representatividad de dominio) |

### Sección 3 — nueva lógica de fusión (lo único de fondo que cambia)
- Descarga **dos** fuentes: base Roboflow v2 (`accident-detection-model`) + fork dashcam del
  usuario (`adrin-cornejo-s-workspace/car-accident-dashcam-j7efi`, v1, 144 imgs).
- **Fusión con dedup MD5 global** procesando `train→valid→test` (first-seen-wins) → garantiza que
  **ninguna imagen quede en dos splits**; los splits se mezclan por split (nunca valid→train).
- **Remap de clases por NOMBRE:** el fork trae 3 clases (`vehicle`, `vehicle-accident`,
  `vehicleaccident`). Cualquier nombre con `accident` → `Accident (0)`; `vehicle` → cajas
  **descartadas** y la imagen se conserva como **negativo** (fondo) → aporta negativos de dominio
  que ayudan a bajar los falsos positivos del baseline.
- **Anti-leakage:** los **8 videos WhatsApp de test NO entran** a ninguna fuente. Held-out total →
  la vara `calibrate_threshold.py` sobre esos 8 videos sigue siendo un test honesto.
- Si el fork es privado, poner la API key propia en `DOMAIN_API_KEY` (por defecto usa la pública).

### Criterio de entrega (sin cambios)
En el repo `ProjectCrashIA1`:
```
venv/Scripts/python tools/calibrate_threshold.py --weights models/best_s_m3.pt --labels tools/labels.csv
```
Manda el recall en los 8 videos (batir **6/8**). Nada se promueve a `config.py` hasta medirlo en video.
