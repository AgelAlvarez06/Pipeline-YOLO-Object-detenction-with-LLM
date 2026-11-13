import json
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from io import BytesIO
from pathlib import Path
 
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
 
import torch
import yaml
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Configuración
SEED = 42
TRAIN_SEQUENCES = 5
VAL_SEQUENCES = 1
 
REPO = "RL-MIND/NJU-HARD-Detection"

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "nju_yolo"
RUNS_DIR = BASE_DIR / "runs_nju"
SEQ_CACHE = DATA_DIR / "seqs_v2.json"  # caché del escaneo de secuencias
 
# "first" : las secuencias que terminan antes en el stream (rápido: el stream
#           se corta en cuanto las tiene todas).
# "random": selección aleatoria con SEED (puede obligar a recorrer todo el split).
SEQUENCE_SELECTION = "first"
 
TILE_SIZE = 1024
OVERLAP = 128
JPEG_QUALITY = 90
 
# Cada worker decodifica un frame completo (~122 MP ≈ 350 MB en RGB).
# Si hay agotamiento de RAM, bajar NUM_WORKERS
NUM_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
MAX_IN_FLIGHT = NUM_WORKERS + 1  # frames en memoria a la vez
 
MAX_RESTARTS = 5  # reaperturas del stream tras un corte de red
 
META_COLUMNS = ["sequence_id", "annotation_status", "objects"]
 
CLASS_NAMES = {
    0: "pedestrian",
    1: "car",
    2: "bus",
    3: "truck",
}
 
Image.MAX_IMAGE_PIXELS = None
 

# Entorno
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
 

# Acceso al dataset (streaming)
def get_stream(split, with_image=True):
    """
    with_image=True : frames completos, imagen como bytes crudos (sin decodificar).
    with_image=False: solo metadatos; no se lee ni decodifica la imagen.
    """
    if with_image:
        ds = load_dataset(REPO, name="default", split=split, streaming=True)
        return ds.cast_column("image", HFImage(decode=False))
 
    try:
        # Con parquet, `columns` evita leer la columna de imagen del disco/red.
        return load_dataset(
            REPO,
            name="default",
            split=split,
            streaming=True,
            columns=META_COLUMNS,
        )
    except TypeError:
        ds = load_dataset(REPO, name="default", split=split, streaming=True)
        return ds.select_columns(META_COLUMNS)
 
 
def is_valid_row(row):
    return row.get("annotation_status") == "present" and row.get("objects") is not None
 

# Escaneo de secuencias (con caché en JSON)
def load_cache():
    if SEQ_CACHE.exists():
        return json.loads(SEQ_CACHE.read_text(encoding="utf-8"))
    return {}
 
 
def scan_sequences(split):
    """
    Devuelve {sequence_id: {"count": n_frames_validos, "last": pos}} donde `pos`
    es la posición en el stream del último frame válido de la secuencia.
    Escanea solo metadatos (sin imágenes) y solo si no está en caché.
    """
    cache = load_cache()
    if split in cache:
        print(f"Secuencias de {split} leídas de caché ({SEQ_CACHE.name})")
        return cache[split]
 
    info = {}
    stream = get_stream(split, with_image=False)
    for index, row in enumerate(tqdm(stream, desc=f"Escaneando {split}", unit="frame")):
        if not is_valid_row(row):
            continue
        seq_id = row.get("sequence_id")
        if seq_id is None:
            continue
        entry = info.setdefault(str(seq_id), {"count": 0, "last": 0})
        entry["count"] += 1
        entry["last"] = index
 
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache[split] = info
    SEQ_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return info
 
 
def get_sequence_counts(split):
    """Devuelve {sequence_id: n_frames_validos}."""
    return {seq: entry["count"] for seq, entry in scan_sequences(split).items()}
 
 
def select_sequence_ids(split, count):
    info = scan_sequences(split)
 
    if len(info) < count:
        raise RuntimeError(
            f"El split {split} tiene {len(info)} secuencias válidas; "
            f"se solicitaron {count}."
        )
 
    if SEQUENCE_SELECTION == "random":
        return sorted(random.sample(sorted(info), count))
 
    # Las `count` secuencias cuyo último frame aparece antes en el stream:
    # así la descarga de imágenes se detiene lo antes posible.
    chosen = sorted(info, key=lambda seq: info[seq]["last"])[:count]
    last_row = max(info[seq]["last"] for seq in chosen)
    n_frames = sum(info[seq]["count"] for seq in chosen)
    print(
        f"[{split}] {n_frames} frames a procesar; el stream se detendrá tras "
        f"recorrer {last_row + 1} frames."
    )
    return sorted(chosen)
 
 
def iter_selected_rows(split, selected_ids):
    """
    Generador: entrega los frames de las secuencias elegidas sin acumularlos en RAM.
    Se detiene en cuanto encuentra todos los frames esperados (conocidos por el escaneo).
    """
    selected = set(selected_ids)
    counts = get_sequence_counts(split)
    expected = sum(counts[s] for s in selected)
    found = 0
 
    bar = tqdm(total=expected, desc=f"Frames {split}", unit="frame")
    try:
        for index, row in enumerate(get_stream(split, with_image=True)):
            # Muestra cuántos frames del stream se han recorrido, para que se
            # vea que sigue vivo aunque no aparezcan frames seleccionados.
            bar.set_postfix(recorridos=index + 1)
 
            if not is_valid_row(row):
                continue
 
            seq_id = row.get("sequence_id")
            if seq_id is not None and str(seq_id) in selected:
                found += 1
                bar.update(1)
                yield row
                if found >= expected:
                    break
    finally:
        bar.close()
 


def tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts
 
 
def process_frame(args):
    """Worker (corre en otro proceso): corta un frame en tiles y guarda imagen + etiquetas."""
    image_bytes, objects, frame_id, split = args
 
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    x_starts = tile_starts(width, TILE_SIZE, OVERLAP)
    y_starts = tile_starts(height, TILE_SIZE, OVERLAP)
 
    # Se parsean los objetos una sola vez, no una vez por tile.
    parsed = []
    for obj in objects:
        category = int(obj["category"])
        if not 0 <= category < len(CLASS_NAMES):
            continue
        x, y, box_w, box_h = map(float, obj["bbox"])
        parsed.append((category, x, y, x + box_w, y + box_h))
 
    saved = 0
 
    for yi, top in enumerate(y_starts):
        for xi, left in enumerate(x_starts):
            right = min(left + TILE_SIZE, width)
            bottom = min(top + TILE_SIZE, height)
            tile_w, tile_h = right - left, bottom - top
            labels = []
 
            for category, x, y, x2, y2 in parsed:
                # Se conservan objetos completos; se omiten los que el tile corta.
                if x < left or y < top or x2 > right or y2 > bottom:
                    continue
 
                cx = ((x + x2) / 2 - left) / tile_w
                cy = ((y + y2) / 2 - top) / tile_h
                nw = (x2 - x) / tile_w
                nh = (y2 - y) / tile_h
                labels.append(f"{category} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
 
            if not labels:
                continue
 
            tile_name = f"{frame_id}_{yi}_{xi}"
            image.crop((left, top, right, bottom)).save(
                DATA_DIR / "images" / split / f"{tile_name}.jpg",
                quality=JPEG_QUALITY,
            )
            (DATA_DIR / "labels" / split / f"{tile_name}.txt").write_text(
                "\n".join(labels) + "\n",
                encoding="utf-8",
            )
            saved += 1
 
    image.close()
    return frame_id, saved
 

# Manifiestos de progreso (para reanudar tras un corte)
def load_manifest(split, kind):
    path = DATA_DIR / f"{kind}_{split}.json"
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return set()
 
 
def save_manifest(split, kind, values):
    path = DATA_DIR / f"{kind}_{split}.json"
    path.write_text(json.dumps(sorted(values)), encoding="utf-8")
 
 
def load_done(split):
    return load_manifest(split, "done")
 
 
def save_done(split, done):
    save_manifest(split, "done", done)
 
 
def load_failed(split):
    return load_manifest(split, "failed")
 
 
def save_failed(split, failed):
    save_manifest(split, "failed", failed)
 
 
def collect_finished(futures, frame_of, split, done, failed):
    """Recoge resultados ya terminados; un frame roto no tumba el pipeline."""
    saved_total = 0
    for future in futures:
        frame_id = frame_of.pop(future, "desconocido")
        try:
            frame_id, saved = future.result()
        except Exception as exc:
            failed.add(frame_id)
            print(f"[{split}] {frame_id}: falló ({type(exc).__name__}: {exc})", flush=True)
            continue
 
        done.add(frame_id)
        failed.discard(frame_id)
        saved_total += saved
        print(f"[{split}] {frame_id}: {saved} tiles", flush=True)
 
    return saved_total
 
 
def tile_split(make_rows_iter, split, max_restarts=MAX_RESTARTS):
    """Consume el generador de frames y reparte el tileado entre procesos.
 
    Reanudable: omite los frames de done_{split}.json.
    Tolerante a cortes de red: si el stream falla, lo reabre y sigue desde el manifiesto.
    """
    total = 0
    done = load_done(split)
    failed = load_failed(split)
    if done:
        print(f"[{split}] Reanudando: {len(done)} frames ya procesados se omitirán.", flush=True)
 
    attempt = 0
    while True:
        pending = set()
        frame_of = {}
        pool = ProcessPoolExecutor(max_workers=NUM_WORKERS)
 
        try:
            for row_index, row in enumerate(make_rows_iter()):
                frame_id = str(row.get("image_id", row_index)).replace("/", "_")
                if frame_id in done:
                    continue
 
                image_data = row.get("image")
                if not image_data or not image_data.get("bytes"):
                    continue
 
                job = (image_data["bytes"], row.get("objects") or [], frame_id, split)
                future = pool.submit(process_frame, job)
                pending.add(future)
                frame_of[future] = frame_id
 
                # Limita los frames en memoria: espera si hay demasiados en curso.
                if len(pending) >= MAX_IN_FLIGHT:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    total += collect_finished(finished, frame_of, split, done, failed)
                    save_done(split, done)
                    save_failed(split, failed)
 
            for future in as_completed(pending):
                total += collect_finished([future], frame_of, split, done, failed)
            save_done(split, done)
            save_failed(split, failed)
 
            pool.shutdown(wait=True)
            return total
 
        except BaseException as exc:
            pool.shutdown(wait=False, cancel_futures=True)
            save_done(split, done)
            save_failed(split, failed)
 
            if isinstance(exc, KeyboardInterrupt):
                raise
 
            attempt += 1
            if attempt > max_restarts:
                raise
 
            wait_s = min(60, 5 * attempt)
            print(
                f"[{split}] Corte en el stream ({type(exc).__name__}: {exc}). "
                f"Reintento {attempt}/{max_restarts} en {wait_s}s "
                f"({len(done)} frames ya guardados).",
                flush=True,
            )
            time.sleep(wait_s)
 
 
def build_yolo_dataset():
    random.seed(SEED)
 
    selected_train_ids = select_sequence_ids("train", TRAIN_SEQUENCES)
    selected_val_ids = select_sequence_ids("validation", VAL_SEQUENCES)
    print("Secuencias train:", selected_train_ids)
    print("Secuencias validation:", selected_val_ids)
 
    for split in ("train", "val"):
        (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
 
    train_tiles = tile_split(lambda: iter_selected_rows("train", selected_train_ids), "train")
    val_tiles = tile_split(lambda: iter_selected_rows("validation", selected_val_ids), "val")
 
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
 

# Revisión del dataset
def check_dataset():
    for split in ("train", "val"):
        images = list((DATA_DIR / "images" / split).glob("*.jpg"))
        labels = list((DATA_DIR / "labels" / split).glob("*.txt"))
        print(f"{split}: {len(images)} imágenes, {len(labels)} etiquetas")
        if not images or not labels:
            raise RuntimeError(f"El split {split} está vacío o no tiene etiquetas.")
 
 
# Entrenamiento, validación y predicción
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
 

# Main
def main():
    mount_drive_if_colab()
    check_gpu()
 
    print("\n[1/4] Descargando frames y generando tiles (en paralelo)...", flush=True)
    build_yolo_dataset()
    check_dataset()
 
    print("\n[2/4] Entrenando...", flush=True)
    results = train_model()
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
 
    print("\n[3/4] Validando...", flush=True)
    best_model = validate_model(best_weights)
 
    print("\n[4/4] Probando una imagen...", flush=True)
    predict_sample(best_model)
 
 
if __name__ == "__main__":
    main()
