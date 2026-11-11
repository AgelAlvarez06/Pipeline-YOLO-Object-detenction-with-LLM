TILE_SIZE = 8192
OVERLAP = 512
CLASSES = ["pedestrian", "car", "bus", "truck"]

# Cajas seguidas que comparten al menos 2 coordenadas (±LOOP_TOL px) cuentan como bucle
LOOP_TOL = 1.5
LOOP_RUN = 4

DEDUP_IOU = 0.5
DEDUP_IOMIN = 0.7


def tile_starts(length, tile_size=TILE_SIZE, overlap=OVERLAP):
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def normalize_box(box):
    x1, x2 = sorted((box["x1"], box["x2"]))
    y1, y2 = sorted((box["y1"], box["y2"]))
    return {**box, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def area(b):
    return max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])


def is_duplicate(a, b):
    w = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    h = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    inter = max(0, w) * max(0, h)
    union = area(a) + area(b) - inter
    smaller = min(area(a), area(b))
    iou = inter / union if union > 0 else 0.0
    iomin = inter / smaller if smaller > 0 else 0.0
    return iou > DEDUP_IOU or iomin > DEDUP_IOMIN


def shared_coords(a, b):
    return sum(abs(a[k] - b[k]) <= LOOP_TOL for k in ("x1", "y1", "x2", "y2"))


def cut_loop(boxes):
    """Quita las cadenas de cajas en bucle, pero conserva las que vienen después."""
    keep, run = [], []
    for b in boxes:
        if run and shared_coords(b, run[-1]) >= 2:
            run.append(b)
            continue
        keep.extend(run if len(run) <= LOOP_RUN else run[:1])
        run = [b]
    keep.extend(run if len(run) <= LOOP_RUN else run[:1])
    return keep


def nearest_start(center, length):
    starts = tile_starts(length)
    return min(starts, key=lambda s: abs(s + min(TILE_SIZE, length - s) / 2 - center))


def owns(box, left, top, width, height):
    """La caja se queda solo en el tile cuyo centro está más cerca del centro de la caja."""
    cx = (box["x1"] + box["x2"]) / 2
    cy = (box["y1"] + box["y2"]) / 2
    return nearest_start(cx, width) == left and nearest_start(cy, height) == top


def clean_tile(record, width, height, stats):
    left, top = record["tile_origin"]
    kept = []
    for category in CLASSES:
        boxes = [normalize_box(b) for b in record.get("boxes_frame_pixels", [])
                 if b.get("category") == category]
        stats["total"] += len(boxes)
        valid = [b for b in boxes if area(b) > 0]
        stats["sin_area"] += len(boxes) - len(valid)
        cut = cut_loop(valid)
        stats["bucle"] += len(valid) - len(cut)
        for b in cut:
            if owns(b, left, top, width, height):
                kept.append(b)
            else:
                stats["solapamiento"] += 1
    return kept


def deduplicate(boxes, stats):
    keep = []
    for b in boxes:
        if any(k["category"] == b["category"] and is_duplicate(k, b) for k in keep):
            stats["duplicadas"] += 1
            continue
        keep.append(b)
    return keep


def clean_frame(records, width, height):
    stats = {"total": 0, "sin_area": 0, "bucle": 0,
             "solapamiento": 0, "duplicadas": 0, "finales": 0}
    boxes = []
    for record in records:
        boxes.extend(clean_tile(record, width, height, stats))
    boxes = deduplicate(boxes, stats)
    stats["finales"] = len(boxes)
    return boxes, stats