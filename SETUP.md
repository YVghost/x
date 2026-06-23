# Guía de instalación — CrashVision M1

**Proyecto:** CrashVision — Detección automática de accidentes de tránsito  
**Modelo:** Re-entrenamiento YOLOv8n / YOLOv8s — 150 epochs  
**Notebook:** `retrain_yolov8_m1.ipynb`

---

## Requisitos del sistema

| Componente | Mínimo | Recomendado |
|---|---|---|
| SO | Windows 10 64-bit | Windows 11 64-bit |
| Python | 3.10 | 3.11.x |
| RAM | 8 GB | 16 GB |
| GPU | NVIDIA (CUDA 11.8+) | RTX 4060+ (8 GB VRAM) |
| Espacio en disco | 10 GB libres | 20 GB libres |
| Driver NVIDIA | 520+ | 610+ |

---

## Paso 1 — Instalar Python 3.11

1. Descarga Python 3.11 desde https://www.python.org/downloads/
2. Durante la instalación, marca **"Add Python to PATH"**
3. Verifica la instalación:

```powershell
python --version
# Python 3.11.x
```

---

## Paso 2 — Clonar / descargar el proyecto

Coloca el proyecto en una ruta sin espacios ni tildes, por ejemplo:

```
C:\Users\<tu_usuario>\proyectos\crashvision\
```

---

## Paso 3 — Crear el entorno virtual

Desde la carpeta raíz del proyecto:

```powershell
python -m venv venv
```

Activa el entorno:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Si PowerShell bloquea scripts, ejecuta primero:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Verifica que el entorno está activo (debe aparecer `(venv)` en el prompt).

---

## Paso 4 — Instalar dependencias base

```powershell
pip install ultralytics>=8.0.0 roboflow>=1.1.0 psutil>=5.9.0 numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 pyyaml>=6.0
```

O con el archivo `requirements.txt`:

```powershell
pip install -r requirements.txt
```

> **Nota:** `requirements.txt` **no incluye** PyTorch con CUDA. Instalarlo por separado (Paso 5).

---

## Paso 5 — Instalar PyTorch con soporte GPU (CUDA)

### Verificar la versión de CUDA del driver

```powershell
nvidia-smi
```

Busca la línea `CUDA UMD Version` y elige el comando correspondiente:

| CUDA del driver | Comando de instalación |
|---|---|
| 11.x | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| 12.6 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| 12.8+ / 13.x | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |

**Este proyecto usa (RTX 4060, driver 610.47, CUDA 13.3):**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Verificar que CUDA fue detectado

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Salida esperada:
```
2.11.0+cu128
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Paso 6 — Configurar el kernel de Jupyter

Para usar el `venv` dentro del notebook en VS Code:

```powershell
pip install ipykernel
python -m ipykernel install --user --name crashvision --display-name "Python (crashvision)"
```

Luego en VS Code:
1. Abre `retrain_yolov8_m1.ipynb`
2. Haz clic en el selector de kernel (esquina superior derecha)
3. Elige **"Python (crashvision)"**

---

## Paso 7 — Estructura de carpetas esperada

Antes de correr el notebook, el proyecto debe tener esta estructura:

```
x\
├── retrain_yolov8_m1.ipynb   ← notebook principal
├── requirements.txt
├── venv\                      ← entorno virtual
└── CrashVision\
    └── models\
        ├── best_n.pt          ← baseline YOLOv8n original (necesario para Sección 6)
        └── best_s.pt          ← baseline YOLOv8s original (necesario para Sección 6)
```

Las carpetas de resultados se crean automáticamente al ejecutar la Sección 1:

```
CrashVision\
├── models\
│   ├── best_n.pt
│   ├── best_s.pt
│   └── m1\
│       ├── best_n_m1.pt       ← generado al terminar Sección 4
│       └── best_s_m1.pt       ← generado al terminar Sección 5
└── results\
    ├── yolov8n_m1\            ← curvas y métricas del modelo n
    ├── yolov8s_m1\            ← curvas y métricas del modelo s
    └── comparison_m1\         ← comparación M1 vs baseline
```

---

## Paso 8 — Ejecutar el notebook

Corre las secciones en orden:

| Sección | Descripción | Tiempo estimado |
|---|---|---|
| 0 | Verificación de entorno y GPU | < 1 min |
| 1 | Configuración de carpetas locales | < 1 min |
| 2 | Verificación de dependencias | < 1 min |
| 3 | Descarga del dataset Roboflow | 1–3 min |
| 4 | Re-entrenamiento YOLOv8n (150 epochs) | 60–90 min (GPU) |
| 5 | Re-entrenamiento YOLOv8s (150 epochs) | 90–150 min (GPU) |
| 6 | Evaluación comparativa | 5–10 min |
| 7 | Visualización de curvas | < 1 min |
| 8 | Exportación de modelos | < 1 min |
| 9 | Resumen y decisión final | < 1 min |

> **Importante:** La Sección 0 debe correrse siempre primero en cada sesión — fija `CUDA_VISIBLE_DEVICES=0` antes de importar torch.

---

## Solución de problemas comunes

### `ModuleNotFoundError: No module named 'google'`
El notebook fue diseñado originalmente para Google Colab. La versión local ya no usa `google.colab`. Si aparece este error, el kernel no está usando el `venv` — revisa el Paso 6.

### `torch.cuda.is_available() → False`
Causas posibles:

1. **PyTorch CPU-only instalado** — reinstala con el comando del Paso 5.
2. **`CUDA_VISIBLE_DEVICES` vacío** — la Sección 0 del notebook ya lo corrige con `os.environ['CUDA_VISIBLE_DEVICES'] = '0'`.
3. **Kernel incorrecto** — verifica que el kernel del notebook sea `Python (crashvision)` y no el Python del sistema.

### `ValueError: ... device=0`
Ocurre si se salta la Sección 0. Reinicia el kernel y corre desde el principio.

### Roboflow descarga lenta o falla
El dataset se descarga una sola vez. Si falla, borra la carpeta del dataset y vuelve a correr la Sección 3.

---

## Resumen de paquetes instalados

| Paquete | Versión instalada | Propósito |
|---|---|---|
| `ultralytics` | 8.4.75 | YOLOv8 — entrenamiento y evaluación |
| `roboflow` | 1.3.10 | Descarga del dataset |
| `torch` | 2.11.0+cu128 | Deep learning con CUDA |
| `torchvision` | 0.26.0+cu128 | Transformaciones de imagen |
| `psutil` | 7.2.2 | Monitoreo de RAM |
| `numpy` | 2.4.6 | Operaciones numéricas |
| `pandas` | 3.0.3 | Análisis de resultados CSV |
| `matplotlib` | 3.11.0 | Visualización de curvas |
| `pyyaml` | 6.0.3 | Lectura de `data.yaml` |
| `opencv-python` | 4.13.0 | Procesamiento de imágenes |
| `scipy` | 1.17.1 | Métricas estadísticas |
| `ipykernel` | — | Integración con VS Code / Jupyter |
