import faulthandler
import json
import os
import re
import sys
import time
from io import BytesIO

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList

# Desactivar la advertencia de "bomba de descompresión" para imágenes grandes
Image.MAX_IMAGE_PIXELS = None

# --- Configuración del piloto ---
DATASET_ID = "RL-MIND/NJU-HARD-Detection"
SPLIT = "train"

MAX_FRAMES = 1  # Empieza con 1 frame para probar
ONLY_ANNOTATED = True

SEED = 42
SELECTION_PATH = f"seleccion_{SPLIT}_seed{SEED}_n{MAX_FRAMES}_v3.json"

TILE_SIZE = 8192
OVERLAP = 512

MAX_NEW_TOKENS = 1024
CACHE_DIR = "cache_frames"
WATCHDOG_SECONDS = 180

OUTPUT_PATH = "predicciones.jsonl"

# Clases anotadas en el dataset.
CLASSES = ["pedestrian", "car", "bus", "truck"]

ROW_COLUMNS = [
    "image", "image_id", "scene", "sequence_id", "frame_id",
    "split", "annotation_status", "objects",
]

BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

fs = HfFileSystem()


# --- Comprobación del entorno ---
if not torch.cuda.is_available():
    raise RuntimeError(
        "No se detectó CUDA. Ejecuta este script en un entorno Linux "
        "con GPU NVIDIA y PyTorch compatible con CUDA."
    )
if torch.cuda.is_bf16_supported():
    COMPUTE_DTYPE = torch.bfloat16
else:
    print("Advertencia: La GPU no reporta soporte BF16. La cuantización usará float16 como respaldo.")
    COMPUTE_DTYPE = torch.float16


# --- Corte de bucles durante la generación ---
class LoopStop(StoppingCriteria):
    def __init__(self, tokenizer, repeats=4, tol=3, every=16):
        self.tokenizer = tokenizer
        self.repeats = repeats
        self.tol = tol
        self.every = every
        self.steps = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.steps += 1
        if self.steps % self.every:
            return False
        text = self.tokenizer.decode(input_ids[0, -300:], skip_special_tokens=False)
        boxes = [tuple(map(int, b)) for b in BOX_PATTERN.findall(text)]
        if len(boxes) <= self.repeats:
            return False
        last = boxes[-(self.repeats + 1):]
        return all(
            max(abs(a - b) for a, b in zip(p, q)) <= self.tol
            for p, q in zip(last, last[1:])
        )


# --- Worker con Cuantización 4-bit usando AutoModel ---
class LocateAnythingWorker:
    """Carga el modelo cuantizado a 4-bit y ejecuta consultas de localización."""

    def __init__(
        self,
        model_path="nvidia/LocateAnything-3B",
        device="cuda",
    ):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

        self.device = device

        # Configuración de cuantización 4-bit (NF4)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map={"": 0},
            torch_dtype=COMPUTE_DTYPE,
        ).eval()

    @torch.no_grad()
    def detect(self, image, categories):
        categories_text = "<​/c>".join(categories)
        question = (
            "Locate all the instances that matches the following "
            f"description: {categories_text}."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.model.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            generation_mode="hybrid",
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([LoopStop(self.tokenizer)]),
            verbose=False,
        )
        answer = response[0] if isinstance(response, tuple) else response
        return answer

    @staticmethod
    def parse_boxes(answer, image_width, image_height):
        """
        Convierte tokens de caja normalizados [0, 1000]
        a coordenadas en píxeles de la imagen recibida.
        """
        boxes = []
        for match in BOX_PATTERN.finditer(str(answer)):
            x1, y1, x2, y2 = [int(value) for value in match.groups()]
            boxes.append(
                {
                    "x1": x1 / 1000 * image_width,
                    "y1": y1 / 1000 * image_height,
                    "x2": x2 / 1000 * image_width,
                    "y2": y2 / 1000 * image_height,
                }
            )
        return boxes


# --- División de imágenes en tiles ---
def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def count_tiles(width, height, tile_size=TILE_SIZE, overlap=OVERLAP):
    return len(tile_starts(height, tile_size, overlap)) * len(tile_starts(width, tile_size, overlap))


def make_tiles(image, tile_size=TILE_SIZE, overlap=OVERLAP):
    width, height = image.size
    for top in tile_starts(height, tile_size, overlap):
        for left in tile_starts(width, tile_size, overlap):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            yield left, top, image.crop((left, top, right, bottom))


def get_pil_image(image_record):
    """Extrae la imagen PIL desde el registro del dataset."""
    image_bytes = image_record.get("bytes")
    if image_bytes is None:
        raise ValueError(
            "La fila no contiene los bytes de la imagen. "
            "Revisa el campo 'image' devuelto por el dataset."
        )
    return Image.open(BytesIO(image_bytes)).convert("RGB")


# --- Selección aleatoria reproducible ---
def open_remote(path):
    return fs.open(path, "rb", cache_type="none")


def row_group_index(pf, row_in_file):
    offset = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if row_in_file < offset + n:
            return rg, row_in_file - offset
        offset += n
    raise IndexError(f"Fila {row_in_file} fuera de rango.")


def select_frames():
    if not os.path.exists(SELECTION_PATH):
        raise RuntimeError(
            f"No existe {SELECTION_PATH}. Restaura tu bloque de selección original para generarlo."
        )
    with open(SELECTION_PATH, "r", encoding="utf-8") as f:
        print(f"Usando selección guardada en {SELECTION_PATH}", flush=True)
        return json.load(f)["selected"]


def read_row(path, row_in_file):
    with open_remote(path) as f:
        pf = pq.ParquetFile(f)
        available = set(pf.schema_arrow.names)
        columns = [c for c in ROW_COLUMNS if c in available]
        rg, local = row_group_index(pf, row_in_file)
        table = pf.read_row_group(rg, columns=columns)
    return table.slice(local, 1).to_pylist()[0]


def load_frame(item):
    os.makedirs(CACHE_DIR, exist_ok=True)
    name = str(item["image_id"]).replace("/", "_")
    img_path = os.path.join(CACHE_DIR, f"{name}.jpg")
    meta_path = os.path.join(CACHE_DIR, f"{name}.json")

    if os.path.exists(img_path) and os.path.exists(meta_path):
        print(f"  Frame en caché: {img_path}", flush=True)
        with open(meta_path, "r", encoding="utf-8") as f:
            row = json.load(f)
        return Image.open(img_path).convert("RGB"), row

    print(f"  Descargando {item['image_id']} (~100 MB)...", flush=True)
    t0 = time.perf_counter()
    row = read_row(item["file"], item["row"])

    if row.get("image_id") != item["image_id"]:
        raise RuntimeError(
            f"{item['file']} fila {item['row']} ya no es {item['image_id']}. "
            f"Borra {SELECTION_PATH} para regenerar la selección."
        )

    image = get_pil_image(row["image"])
    with open(img_path, "wb") as f:
        f.write(row["image"]["bytes"])
    meta = {k: v for k, v in row.items() if k != "image"}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, default=str)

    print(f"  Descarga terminada en {time.perf_counter() - t0:.1f}s", flush=True)
    return image, meta


# --- Ejecución del pipeline ---
def main():
    from transformers import set_seed

    set_seed(SEED)

    t0 = time.perf_counter()
    selected = select_frames()
    print(f"[tiempo] Selección: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    worker = LocateAnythingWorker()
    print(f"[tiempo] Carga del modelo: {time.perf_counter() - t0:.1f}s", flush=True)

    processed_frames = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        for item in selected:
            t0 = time.perf_counter()
            image, row = load_frame(item)
            print(f"[tiempo] Lectura del frame: {time.perf_counter() - t0:.1f}s", flush=True)

            frame_width, frame_height = image.size
            total_tiles = count_tiles(frame_width, frame_height)
            print(
                f"Procesando frame {processed_frames + 1}/{len(selected)}: "
                f"{row.get('image_id')} ({frame_width}x{frame_height}), "
                f"{total_tiles} tiles x {len(CLASSES)} clases",
                flush=True,
            )

            frame_start = time.perf_counter()

            for tile_index, (left, top, tile) in enumerate(make_tiles(image)):
                boxes = []
                raw_answers = {}
                tile_start = time.perf_counter()

                for category in CLASSES:
                    t0 = time.perf_counter()
                    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, file=sys.stderr)
                    answer = worker.detect(tile, [category])
                    faulthandler.cancel_dump_traceback_later()
                    elapsed = time.perf_counter() - t0

                    raw_answers[category] = str(answer)
                    category_boxes = worker.parse_boxes(answer, tile.width, tile.height)
                    print(
                        f"    {category}: {len(category_boxes)} cajas, {elapsed:.1f}s",
                        flush=True,
                    )

                    for box in category_boxes:
                        # Convertir coordenadas del tile al frame original
                        box["x1"] = max(0, min(frame_width, box["x1"] + left))
                        box["y1"] = max(0, min(frame_height, box["y1"] + top))
                        box["x2"] = max(0, min(frame_width, box["x2"] + left))
                        box["y2"] = max(0, min(frame_height, box["y2"] + top))
                        box["category"] = category

                    boxes.extend(category_boxes)

                result = {
                    "image_id": row.get("image_id"),
                    "source_file": item["file"],
                    "row_in_file": item["row"],
                    "seed": SEED,
                    "scene": row.get("scene"),
                    "sequence_id": row.get("sequence_id"),
                    "frame_id": row.get("frame_id"),
                    "split": row.get("split"),
                    "annotation_status": row.get("annotation_status"),
                    "tile_index": tile_index,
                    "tile_origin": [left, top],
                    "raw_answers": raw_answers,
                    "boxes_frame_pixels": boxes,
                    "ground_truth": row.get("objects"),
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()

                done = tile_index + 1
                avg = (time.perf_counter() - frame_start) / done
                remaining = avg * (total_tiles - done) / 60
                print(
                    f"  Tile {done}/{total_tiles}: {len(boxes)} cajas, "
                    f"{time.perf_counter() - tile_start:.1f}s "
                    f"(restante aprox. {remaining:.1f} min)",
                    flush=True,
                )

            image.close()
            processed_frames += 1
            print(
                f"[tiempo] Frame completo: {(time.perf_counter() - frame_start) / 60:.1f} min",
                flush=True,
            )

    faulthandler.cancel_dump_traceback_later()
    print(f"Listo. Frames procesados: {processed_frames}")
    print(f"Resultados guardados en: {OUTPUT_PATH}")
    print(f"Selección usada: {SELECTION_PATH}")


if __name__ == "__main__":
    main()