"""
Genera una imagen de ejemplo con las cajas predichas y su confianza en %.
 
Usa el modelo ya entrenado (no vuelve a entrenar). Ejecutar desde la misma
carpeta que proyecto_YOLO.py:
 
    python predecir_ejemplo.py                # elige sola una imagen de validación
    python predecir_ejemplo.py ruta/imagen.jpg  # o usa la imagen que le indiques
"""
 
import sys
from pathlib import Path
 
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
 
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "nju_yolo"
RUNS_DIR = BASE_DIR / "runs_nju"
OUTPUT = BASE_DIR / "prediccion_ejemplo.jpg"
 
CONF = 0.25        # confianza mínima para dibujar una caja
IMGSZ = 640        # mismo tamaño que en el entrenamiento
MAX_CANDIDATES = 30  # imágenes de validación a probar como máximo
 
COLORS = ["#FF3838", "#18A0FB", "#FFB21D", "#2ECC71"]
 
 
def find_weights():
    """Devuelve el best.pt más reciente dentro de runs_nju/."""
    candidates = sorted(
        RUNS_DIR.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No se encontró ningún best.pt en {RUNS_DIR}")
    return candidates[-1]
 
 
def candidate_images():
    """Imágenes de validación ordenadas de más a menos objetos etiquetados."""
    counted = []
    for label_path in (DATA_DIR / "labels" / "val").glob("*.txt"):
        image_path = DATA_DIR / "images" / "val" / f"{label_path.stem}.jpg"
        if image_path.exists():
            n_objects = len(label_path.read_text(encoding="utf-8").splitlines())
            counted.append((n_objects, image_path))
    counted.sort(key=lambda item: (-item[0], item[1].name))
    return [path for _, path in counted[:MAX_CANDIDATES]]
 
 
def load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()
 
 
def draw_predictions(image_path, result):
    """Dibuja cada caja con 'clase NN%' y devuelve la imagen."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(max(12, image.width // 70))
    line_width = max(2, image.width // 400)
 
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        color = COLORS[class_id % len(COLORS)]
        text = f"{result.names[class_id]} {confidence * 100:.0f}%"
 
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
 
        # Etiqueta encima de la caja (o dentro si no cabe arriba).
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        label_y = y1 - text_h - 4 if y1 - text_h - 4 >= 0 else y1
        draw.rectangle((x1, label_y, x1 + text_w + 6, label_y + text_h + 4), fill=color)
        draw.text((x1 + 3, label_y + 2 - top), text, fill="white", font=font)
 
    return image
 
 
def main():
    weights = find_weights()
    print("Modelo:", weights)
    model = YOLO(str(weights))
 
    if len(sys.argv) > 1:
        images = [Path(sys.argv[1])]
    else:
        images = candidate_images()
        if not images:
            raise RuntimeError("No hay imágenes de validación con etiquetas.")
 
    best_path, best_result = None, None
    for image_path in images:
        result = model.predict(
            source=str(image_path), imgsz=IMGSZ, conf=CONF, device=0, verbose=False
        )[0]
        if best_result is None or len(result.boxes) > len(best_result.boxes):
            best_path, best_result = image_path, result
 
    if len(best_result.boxes) == 0:
        print(
            f"Ninguna de las {len(images)} imágenes probadas tuvo detecciones "
            f"con conf >= {CONF}. Prueba a bajar CONF (por ejemplo a 0.10)."
        )
        return
 
    draw_predictions(best_path, best_result).save(OUTPUT, quality=95)
 
    print("Imagen original:", best_path)
    print(f"Detecciones: {len(best_result.boxes)}")
    for box in sorted(best_result.boxes, key=lambda b: -float(b.conf[0])):
        name = best_result.names[int(box.cls[0])]
        print(f"  {name}: {float(box.conf[0]) * 100:.1f}%")
    print("Imagen con predicciones guardada en:", OUTPUT)
 
 
if __name__ == "__main__":
    main()