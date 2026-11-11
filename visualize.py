import json
import os

from PIL import Image, ImageDraw, ImageFont

from postproceso import clean_frame

Image.MAX_IMAGE_PIXELS = None

# --- Configuración ---
JSONL_PATH = "predicciones.jsonl"
CACHE_DIR = "cache_frames"
SCALE = 0.2

# IDs de las clases del dataset (0: pedestrian, 1: car, 2: bus, 3: truck)
CLASS_NAMES = {0: "pedestrian", 1: "car", 2: "bus", 3: "truck"}


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_label(draw, x, y, text, color, font, below=False):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left, bottom - top + 4
    y = y + 2 if below else max(0, y - h)
    draw.rectangle([x, y, x + w + 4, y + h], fill=color)
    draw.text((x + 2, y), text, fill="white", font=font)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_iou(box, gt_boxes):
    pred = (box["x1"], box["y1"], box["x2"], box["y2"])
    candidates = [g["xyxy"] for g in gt_boxes if g["category"] == box.get("category")]
    return max((iou(pred, g) for g in candidates), default=0.0)


def visualize_predictions(image_id, jsonl_path, output_name="visualizacion.jpg"):
    print(f"Buscando imagen {image_id} en {CACHE_DIR}...")

    # 1. Cargar la imagen original y el ground truth desde la caché del pipeline
    name = str(image_id).replace("/", "_")
    img_path = os.path.join(CACHE_DIR, f"{name}.jpg")
    meta_path = os.path.join(CACHE_DIR, f"{name}.json")

    if not (os.path.exists(img_path) and os.path.exists(meta_path)):
        print("Imagen no encontrada en la caché. Ejecuta primero pipeline.py.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        gt_objects = json.load(f).get("objects") or []

    gt_boxes = []
    for obj in gt_objects:
        x, y, w, h = obj["bbox"]
        gt_boxes.append({
            "category": CLASS_NAMES.get(obj["category"], "unknown"),
            "xyxy": (x, y, x + w, y + h),
        })

    # 2. Leer las predicciones del archivo JSONL
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["image_id"] == image_id:
                records.append({
                    "tile_origin": data["tile_origin"],
                    "boxes_frame_pixels": data.get("boxes_frame_pixels", []),
                })

    # 3. Redimensionar antes de dibujar (122MP es demasiado)
    with Image.open(img_path) as img:
        width, height = img.size
        new_size = (int(width * SCALE), int(height * SCALE))
        img.draft("RGB", new_size)
        image_pil = img.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)

    predictions, stats = clean_frame(records, width, height)
    print(f"Postproceso: {stats}")

    # 4. Dibujar las cajas
    draw = ImageDraw.Draw(image_pil)
    font = load_font(14)
    s = SCALE

    # Dibujar Ground Truth (Verde)
    for obj in gt_objects:
        x, y, w, h = [v * s for v in obj["bbox"]]
        draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
        category = CLASS_NAMES.get(obj["category"], "unknown")
        draw_label(draw, x, y, f"GT: {category}", "green", font)

    # Dibujar Predicciones (Rojo) con IoU respecto al GT más cercano de la misma clase
    for box in predictions:
        x1, y1, x2, y2 = box["x1"] * s, box["y1"] * s, box["x2"] * s, box["y2"] * s
        score = best_iou(box, gt_boxes)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw_label(
            draw, x1, y2,
            f"Pred: {box.get('category', '?')} {score:.0%}",
            "red", font, below=True,
        )

    # 5. Guardar
    image_pil.save(output_name, quality=90)
    print(f"GT: {len(gt_objects)} cajas, predicciones: {len(predictions)} cajas")
    print(f"Imagen guardada como {output_name}")


if __name__ == "__main__":
    # Cambia esto por el image_id que quieras visualizar
    TARGET_IMAGE_ID = None
    if TARGET_IMAGE_ID is None:
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            TARGET_IMAGE_ID = json.loads(f.readline())["image_id"]
    visualize_predictions(TARGET_IMAGE_ID, JSONL_PATH)