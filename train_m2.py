"""
M2 — Re-entrenamiento YOLOv8s con DATA AUGMENTATION dirigido.
==============================================================

Contexto (ver HANDOFF §0bis y memoria del proyecto):
  - M1 ("más épocas + early stopping") mejoró el test set Roboflow (F1 0.873) pero
    REGRESÓ en video real (recall 75% -> 50% @ conf=0.30). Optimizar la distribución
    conocida no arregla la generalización.
  - M2 ataca el cuello de botella real: expone al modelo a variaciones que aproximan
    el video de producción (iluminación/noche, clima, ángulo dashcam vs vigilancia,
    distancia, oclusión) vía augmentation, manteniendo la receta de M1.

Solo se entrena 's' (YOLOv8n se descartó en M1 por regresión de recall).

Ejecutar en la laptop GPU (RTX 4060, entorno descrito en SETUP.md):
    python train_m2.py

Salida: runs/detect/crashvision_s_m2/weights/best.pt  (-> best_s_m2.pt)
Luego validar con el MISMO criterio que M1:
    # en el repo ProjectCrashIA1, tras copiar best_s_m2.pt a models/
    venv/Scripts/python tools/calibrate_threshold.py --weights models/best_s_m2.pt --labels tools/labels.csv
La métrica que decide la entrega es el recall en VIDEO real (batir el baseline 75%).
"""
import os
# Forzar GPU 0 antes de importar torch (igual que el notebook M1).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
from pathlib import Path

import torch
from roboflow import Roboflow
from ultralytics import YOLO

# --- Receta heredada de M1 (sin cambios) -----------------------------------
EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"
MODEL_NAME = "crashvision_s_m2"

# --- Dataset Roboflow v2 (idéntico a M1: misma distribución, cambia el augm.)
ROBOFLOW_API_KEY = "DwCmQ00XOYIzR0pfSFUK"
WORKSPACE_SLUG = "accident-detection-model"
PROJECT_SLUG = "accident-detection-model"
DATASET_VERSION = 2


def get_data_yaml() -> str:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE_SLUG).project(PROJECT_SLUG)
    dataset = project.version(DATASET_VERSION).download("yolov8")
    return str(Path(dataset.location) / "data.yaml")


def main():
    if DEVICE == "cpu":
        print("ADVERTENCIA: CUDA no detectado. M2 en CPU es inviable (horas). Aborta y "
              "corre esto en la laptop con RTX 4060.")
    print("=" * 64)
    print("  RE-ENTRENAMIENTO YOLOv8s M2 — CrashVision (augmentation dirigido)")
    print("=" * 64)
    print(f"  epochs={EPOCHS} | patience={PATIENCE} | batch={BATCH_SIZE} | imgsz={IMG_SIZE}")
    print(f"  device={DEVICE} | name={MODEL_NAME}")
    print("=" * 64)

    data_yaml = get_data_yaml()
    model = YOLO("yolov8s.pt")  # base COCO, entrenamiento desde cero (igual que M1)

    t0 = time.time()
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        name=MODEL_NAME,
        exist_ok=True,
        workers=2,
        cache=False,
        save=True,
        plots=True,
        verbose=True,
        # --- M2: AUGMENTATION dirigido a los modos de fallo del VIDEO real ---
        # Iluminación / clima (noche, baja luz, días grises, lluvia):
        hsv_h=0.020,        # tono   (default 0.015)
        hsv_s=0.80,         # satur. (default 0.70)
        hsv_v=0.60,         # brillo (default 0.40)  <- clave para noche/baja luz
        # Geometría / punto de vista (dashcam vs cámara de vigilancia elevada):
        degrees=10.0,       # rotación        (default 0.0)
        translate=0.15,     # desplazamiento  (default 0.10)
        scale=0.6,          # escala/distancia(default 0.50)
        shear=3.0,          # cizalla         (default 0.0)
        perspective=0.0005, # perspectiva     (default 0.0)
        fliplr=0.5,         # espejo horizontal (default 0.5)
        # Composición y oclusión:
        mosaic=1.0,         # (default 1.0)
        close_mosaic=10,    # apaga mosaic las últimas 10 epochs para estabilizar
        mixup=0.15,         # mezcla de imágenes (default 0.0)
        copy_paste=0.10,    # copiar-pegar instancias (default 0.0)
        erasing=0.4,        # random erasing / oclusión parcial (default 0.4)
    )
    print(f"\nYOLOv8s M2 completado en {(time.time() - t0) / 60:.1f} minutos")

    best = Path(f"runs/detect/{MODEL_NAME}/weights/best.pt")
    print(f"Mejor modelo: {best}  (cópialo como best_s_m2.pt y valídalo en los 8 videos)")


if __name__ == "__main__":
    main()
