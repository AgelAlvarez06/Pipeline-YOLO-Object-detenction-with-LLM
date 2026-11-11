"""
Proyecto YOLO: detección en NJU-HARD-Detection con tiles.

Dependencias (instalar antes de ejecutar):
    pip install ultralytics datasets pillow pyyaml torch
"""

import random
from io import BytesIO
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
SEED = 42
TRAIN_SEQUENCES = 5
VAL_SEQUENCES = 1

REPO = "RL-MIND/NJU-HARD-Detection"

# En Colab puedes cambiar BASE_DIR a Path("/content")
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "nju_yolo"
RUNS_DIR = BASE_DIR / "runs_nju"

TILE_SIZE = 1024
OVERLAP = 128

CLASS_NAMES = {
    0: "pedestrian",
    1: "car",
    2: "bus",
    3: "truck",
}

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------
# Entorno
# --------------------------------------------------------------------------
def mount_drive_if_colab():
    """Monta Google Drive solo si se ejecuta en Colab (opcional)."""
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except ImportError:
        pass


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No hay GPU disponible. En Colab ve a Entorno de ejecución → "
            "Cambiar tipo de entorno de ejecución → GPU."
        )
    print(torch.cuda.get_device_name(0))


# --------------------------------------------------------------------------
# Descarga de muestra reproducible
# --------------------------------------------------------------------------
def get_stream(split):
    return load_dataset(
        REPO,
        name="default",
        split=split,
        streaming=True,
    )


def is_valid_row(row):
    return row.get("annotation_status") == "present" and row.get("objects") is not None


def select_sequence_ids(split, count):
    sequence_ids = set()

    for row in tqdm(get_stream(split), desc=f"Escaneando {split}", unit="frame"):
        if not is_valid_row(row):
            continue

        seq_id = row.get("sequence_id")
        if seq_id is not None:
            sequence_ids.add(str(seq_id))

    sequence_ids = sorted(sequence_ids)

    if len(sequence_ids) < count:
        raise RuntimeError(
            f"El split {split} tiene {len(sequence_ids)} secuencias válidas; "
            f"se solicitaron {count}."
        )

    return sorted(random.sample(sequence_ids, count))


def load_selected_rows(split, selected_ids):
    selected_ids = set(selected_ids)
    rows = []

    for row in tqdm(get_stream(split), desc=f"Cargando {split}", unit="frame"):
        if not is_valid_row(row):
            continue

        seq_id = row.get("sequence_id")
        if seq_id is not None and str(seq_id) in selected_ids:
            rows.append(row)

    return rows


def download_sample():
    random.seed(SEED)

    selected_train_ids = select_sequence_ids("train", TRAIN_SEQUENCES)
    selected_val_ids = select_sequence_ids("validation", VAL_SEQUENCES)

    train_rows = load_selected_rows("train", selected_train_ids)
    val_rows = load_selected_rows("validation", selected_val_ids)

    print("Secuencias train:", selected_train_ids)
    print("Secuencias validation:", selected_val_ids)
    print("Frames seleccionados:", len(train_rows), "train;", len(val_rows), "validation")

    return train_rows, val_rows


# --------------------------------------------------------------------------
# Tiles y etiquetas en formato YOLO
# --------------------------------------------------------------------------
def tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def save_tiles(rows, split):
    image_count = 0

    for row_index, row in enumerate(tqdm(rows, desc=f"Tiles {split}", unit="frame")):
        image_data = row.get("image")
        if not image_data or not image_data.get("bytes"):
            continue

        image = Image.open(BytesIO(image_data["bytes"])).convert("RGB")
        width, height = image.size
        x_starts = tile_starts(width, TILE_SIZE, OVERLAP)
        y_starts = tile_starts(height, TILE_SIZE, OVERLAP)

        objects = row.get("objects") or []
        frame_id = str(row.get("image_id", row_index)).replace("/", "_")

        for yi, top in enumerate(y_starts):
            for xi, left in enumerate(x_starts):
                right = min(left + TILE_SIZE, width)
                bottom = min(top + TILE_SIZE, height)
                tile = image.crop((left, top, right, bottom))
                tile_w, tile_h = tile.size
                labels = []

                for obj in objects:
                    category = int(obj["category"])
                    x, y, box_w, box_h = map(float, obj["bbox"])
                    x2, y2 = x + box_w, y + box_h

                    ix1 = max(x, left)
                    iy1 = max(y, top)
                    ix2 = min(x2, right)
                    iy2 = min(y2, bottom)

                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    # Se conservan objetos completos; se omiten los que el tile corta.
                    if ix1 > x or iy1 > y or ix2 < x2 or iy2 < y2:
                        continue

                    cx = ((ix1 + ix2) / 2 - left) / tile_w
                    cy = ((iy1 + iy2) / 2 - top) / tile_h
                    nw = (ix2 - ix1) / tile_w
                    nh = (iy2 - iy1) / tile_h

                    if 0 <= category < len(CLASS_NAMES):
                        labels.append(
                            f"{category} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                        )

                if not labels:
                    continue

                tile_name = f"{frame_id}_{yi}_{xi}"
                tile.save(DATA_DIR / "images" / split / f"{tile_name}.jpg", quality=95)
                (DATA_DIR / "labels" / split / f"{tile_name}.txt").write_text(
                    "\n".join(labels) + "\n",
                    encoding="utf-8",
                )
                image_count += 1

        image.close()

    return image_count


def build_yolo_dataset(train_rows, val_rows):
    for split in ("train", "val"):
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_tiles = save_tiles(train_rows, "train")
    val_tiles = save_tiles(val_rows, "val")

    if train_tiles == 0 or val_tiles == 0:
        raise RuntimeError(
            f"No se generaron suficientes tiles etiquetados: "
            f"{train_tiles} train, {val_tiles} validation."
        )

    data_yaml = {
        "path": str(DATA_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }

    with open(DATA_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"Tiles etiquetados: {train_tiles} train; {val_tiles} validation")
    print(f"Configuración guardada en: {DATA_DIR / 'data.yaml'}")


# --------------------------------------------------------------------------
# Revisión del dataset
# --------------------------------------------------------------------------
def check_dataset():
    for split in ("train", "val"):
        images = list((DATA_DIR / "images" / split).glob("*.jpg"))
        labels = list((DATA_DIR / "labels" / split).glob("*.txt"))
        print(f"{split}: {len(images)} imágenes, {len(labels)} etiquetas")
        if not images or not labels:
            raise RuntimeError(f"El split {split} está vacío o no tiene etiquetas.")


# --------------------------------------------------------------------------
# Entrenamiento, validación y predicción
# --------------------------------------------------------------------------
def train_model():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATA_DIR / "data.yaml"),
        epochs=2,
        imgsz=640,
        batch=2,
        workers=2,
        device=0,
        project=str(RUNS_DIR),
        name="yolo11n_piloto",
        patience=8,
        save=True,
        plots=True,
    )
    return results


def validate_model(best_weights):
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(DATA_DIR / "data.yaml"),
        imgsz=640,
        batch=4,
        device=0,
    )

    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")

    return best_model


def predict_sample(best_model):
    val_images = sorted((DATA_DIR / "images" / "val").glob("*.jpg"))
    if not val_images:
        raise RuntimeError("No hay imágenes de validación para probar.")

    best_model.predict(
        source=str(val_images[0]),
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
    )

    print("Imagen de prueba:", val_images[0])
    print("Resultado anotado en runs/detect/predict")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    mount_drive_if_colab()
    check_gpu()

    print("\n[1/5] Descargando muestra (puede tardar)...", flush=True)
    train_rows, val_rows = download_sample()

    print("\n[2/5] Generando tiles y etiquetas...", flush=True)
    build_yolo_dataset(train_rows, val_rows)
    check_dataset()

    print("\n[3/5] Entrenando...", flush=True)
    results = train_model()
    best_weights = Path(results.save_dir) / "weights" / "best.pt"

    print("\n[4/5] Validando...", flush=True)
    best_model = validate_model(best_weights)

    print("\n[5/5] Probando una imagen...", flush=True)
    predict_sample(best_model)

if __name__ == "__main__":
    main()