"""
Fase 1: ejecuta YOLO y guarda las detecciones como un JSON por frame.

Uso:
    python -m agente.exportar_detecciones --fuente nju_yolo/images/val
    python -m agente.exportar_detecciones --fuente frame.jpg --pesos best.pt

Modos:
- frames: cada imagen es un frame completo. Si es mayor que TILE_SIZE se corta
  en tiles solapados, se predice cada tile, las cajas se pasan a coordenadas del
  frame y se deduplican con NMS por clase (DEDUP_IOU).
- tiles: imágenes generadas por yolo.py ({frame_id}_{fila}_{columna}.jpg). Se
  reagrupan por frame. Como yolo.py solo guarda tiles con etiquetas y no guarda el
  tamaño del frame, la posición global es aproximada y la cobertura es parcial.
- auto (por defecto): usa "tiles" si todas las imágenes siguen ese patrón y no
  superan TILE_SIZE; si no, "frames".
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from agente.config import CLASS_NAMES, RAIZ, Config, cargar_config

Image.MAX_IMAGE_PIXELS = None  # los frames de NJU-HARD superan el límite de Pillow

EXTENSIONES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PATRON_TILE = re.compile(r"^(?P<frame>.+)_(?P<fila>\d+)_(?P<col>\d+)$")
LOTE = 8  # tiles por llamada a model.predict


# ---------------------------------------------------------------------------
# Funciones puras (testeables sin modelo)
# ---------------------------------------------------------------------------
def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Mismo algoritmo de corte que yolo.py."""
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def celda_de(cx: float, cy: float, ancho: float, alto: float, filas: int, columnas: int) -> str:
    """Celda 'rXcY' de la cuadrícula filas x columnas donde cae el punto (cx, cy)."""
    fila = min(filas - 1, max(0, int(cy / alto * filas))) if alto > 0 else 0
    col = min(columnas - 1, max(0, int(cx / ancho * columnas))) if ancho > 0 else 0
    return f"r{fila}c{col}"


def inferir_split(ruta: Path) -> str:
    partes = {p.lower() for p in ruta.parts}
    if partes & {"val", "validation"}:
        return "val"
    if "train" in partes:
        return "train"
    if "test" in partes:
        return "test"
    return "desconocido"


def construir_registro(
    *,
    frame_id: str,
    detecciones: list[dict],
    ancho: int,
    alto: int,
    ruta_imagen: str,
    split: str,
    secuencia: str | None,
    modelo: dict,
    config: Config,
    coordenadas: str = "exactas",
    cobertura: str = "completa",
    n_tiles: int | None = None,
) -> dict:
    """Arma el JSON del frame: resumen precalculado + lista de detecciones."""
    filas, cols = config.grid_filas, config.grid_columnas
    clases = list(CLASS_NAMES.values())
    for det in detecciones:
        if det["clase"] not in clases:
            clases.append(det["clase"])

    por_clase = Counter({c: 0 for c in clases})
    suma_conf: dict[str, float] = defaultdict(float)
    por_celda: dict[str, Counter] = defaultdict(Counter)

    salida = []
    orden = sorted(detecciones, key=lambda d: -d["confianza"])
    for i, det in enumerate(orden):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        celda = celda_de((x1 + x2) / 2, (y1 + y2) / 2, ancho, alto, filas, cols)
        por_clase[det["clase"]] += 1
        suma_conf[det["clase"]] += det["confianza"]
        por_celda[celda][det["clase"]] += 1
        salida.append(
            {
                "id": i,
                "clase": det["clase"],
                "clase_id": det["clase_id"],
                "confianza": round(det["confianza"], 4),
                "bbox_xyxy": [round(v, 1) for v in det["bbox_xyxy"]],
                "celda": celda,
            }
        )

    return {
        "frame_id": frame_id,
        "secuencia": secuencia,
        "split": split,
        "imagen": {"ruta": ruta_imagen, "ancho": ancho, "alto": alto},
        "modelo": modelo,
        "procesado_en": datetime.now().isoformat(timespec="seconds"),
        "cobertura": {"tipo": cobertura, "coordenadas": coordenadas, "n_tiles": n_tiles},
        "grid": {"filas": filas, "columnas": cols},
        "resumen": {
            "total": len(salida),
            "por_clase": dict(por_clase),
            "confianza_media_por_clase": {
                c: round(suma_conf[c] / por_clase[c], 4) for c in clases if por_clase[c]
            },
            "conteo_por_celda": {
                celda: dict(conteo) for celda, conteo in sorted(por_celda.items())
            },
        },
        "detecciones": salida,
    }


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def encontrar_pesos(config: Config, pesos: str | None) -> Path:
    if pesos:
        ruta = Path(pesos).expanduser()
        return ruta if ruta.is_absolute() or not (RAIZ / ruta).exists() else RAIZ / ruta
    if config.yolo_weights:
        if not config.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO_WEIGHTS no existe: {config.yolo_weights}")
        return config.yolo_weights
    candidatos = sorted(
        config.runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime
    )
    if not candidatos:
        raise FileNotFoundError(
            f"No hay best.pt en {config.runs_dir}. Copia el modelo entrenado en Colab "
            "ahí, define YOLO_WEIGHTS en .env o usa --pesos."
        )
    return candidatos[-1]


def elegir_dispositivo():
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cajas(resultado, dx: float = 0.0, dy: float = 0.0) -> list[dict]:
    """Convierte un Results de ultralytics en dicts, desplazando al frame."""
    nombres = resultado.names
    cajas = []
    for xyxy, conf, cls in zip(
        resultado.boxes.xyxy.tolist(),
        resultado.boxes.conf.tolist(),
        resultado.boxes.cls.tolist(),
    ):
        x1, y1, x2, y2 = xyxy
        clase_id = int(cls)
        cajas.append(
            {
                "clase_id": clase_id,
                "clase": nombres[clase_id],
                "confianza": float(conf),
                "bbox_xyxy": [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
            }
        )
    return cajas


def deduplicar(cajas: list[dict], iou: float) -> list[dict]:
    """NMS por clase para eliminar cajas repetidas en zonas de solapamiento."""
    if not cajas:
        return []
    import torch
    from torchvision.ops import batched_nms

    boxes = torch.tensor([c["bbox_xyxy"] for c in cajas], dtype=torch.float32)
    scores = torch.tensor([c["confianza"] for c in cajas], dtype=torch.float32)
    clases = torch.tensor([c["clase_id"] for c in cajas], dtype=torch.int64)
    keep = batched_nms(boxes, scores, clases, iou).tolist()
    return [cajas[i] for i in keep]


def _predecir(model, imagenes, config: Config, device) -> list:
    return model.predict(
        source=imagenes,
        imgsz=config.yolo_imgsz,
        conf=config.yolo_conf,
        device=device,
        verbose=False,
    )


def predecir_frame(model, imagen: Image.Image, config: Config, device) -> list[dict]:
    ancho, alto = imagen.size
    xs = tile_starts(ancho, config.tile_size, config.tile_overlap)
    ys = tile_starts(alto, config.tile_size, config.tile_overlap)
    posiciones = [(x, y) for y in ys for x in xs]

    cajas: list[dict] = []
    for i in range(0, len(posiciones), LOTE):
        lote = posiciones[i : i + LOTE]
        recortes = [
            imagen.crop((x, y, min(x + config.tile_size, ancho), min(y + config.tile_size, alto)))
            for x, y in lote
        ]
        for (x, y), resultado in zip(lote, _predecir(model, recortes, config, device)):
            cajas.extend(_cajas(resultado, x, y))

    return deduplicar(cajas, config.dedup_iou) if len(posiciones) > 1 else cajas


# ---------------------------------------------------------------------------
# Recorrido de fuentes
# ---------------------------------------------------------------------------
def listar_imagenes(fuente: Path) -> list[Path]:
    if fuente.is_file():
        return [fuente]
    if not fuente.is_dir():
        raise FileNotFoundError(f"No existe la fuente: {fuente}")
    imagenes = sorted(p for p in fuente.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not imagenes:
        raise FileNotFoundError(f"No hay imágenes en {fuente}")
    return imagenes


def detectar_modo(imagenes: list[Path], config: Config) -> str:
    for ruta in imagenes:
        if not PATRON_TILE.match(ruta.stem):
            return "frames"
        with Image.open(ruta) as img:
            if max(img.size) > config.tile_size:
                return "frames"
    return "tiles"


def exportar_frames(model, imagenes, config, device, modelo_info, secuencia, destino) -> list[Path]:
    rutas = []
    for ruta in imagenes:
        with Image.open(ruta) as img:
            imagen = img.convert("RGB")
        cajas = predecir_frame(model, imagen, config, device)
        ancho, alto = imagen.size
        n_tiles = len(tile_starts(ancho, config.tile_size, config.tile_overlap)) * len(
            tile_starts(alto, config.tile_size, config.tile_overlap)
        )
        imagen.close()
        registro = construir_registro(
            frame_id=ruta.stem,
            detecciones=cajas,
            ancho=ancho,
            alto=alto,
            ruta_imagen=_relativa(ruta),
            split=inferir_split(ruta),
            secuencia=secuencia,
            modelo=modelo_info,
            config=config,
            n_tiles=n_tiles,
        )
        rutas.append(_guardar(registro, destino))
    return rutas


def exportar_tiles(model, imagenes, config, device, modelo_info, secuencia, destino) -> list[Path]:
    paso = config.tile_size - config.tile_overlap
    grupos: dict[str, list[tuple[Path, int, int]]] = defaultdict(list)
    for ruta in imagenes:
        m = PATRON_TILE.match(ruta.stem)
        grupos[m["frame"]].append((ruta, int(m["fila"]), int(m["col"])))

    rutas = []
    for frame_id, tiles in sorted(grupos.items()):
        cajas: list[dict] = []
        ancho = alto = 0
        for i in range(0, len(tiles), LOTE):
            lote = tiles[i : i + LOTE]
            resultados = _predecir(model, [str(t[0]) for t in lote], config, device)
            for (ruta, fila, col), resultado in zip(lote, resultados):
                dx, dy = col * paso, fila * paso
                alto_t, ancho_t = resultado.orig_shape
                ancho, alto = max(ancho, dx + ancho_t), max(alto, dy + alto_t)
                cajas.extend(_cajas(resultado, dx, dy))
        registro = construir_registro(
            frame_id=frame_id,
            detecciones=deduplicar(cajas, config.dedup_iou),
            ancho=ancho,
            alto=alto,
            ruta_imagen=_relativa(tiles[0][0].parent),
            split=inferir_split(tiles[0][0]),
            secuencia=secuencia,
            modelo=modelo_info,
            config=config,
            coordenadas="aproximadas",
            cobertura="parcial (solo tiles con etiquetas guardados por yolo.py)",
            n_tiles=len(tiles),
        )
        rutas.append(_guardar(registro, destino))
    return rutas


def _relativa(ruta: Path) -> str:
    try:
        return str(ruta.resolve().relative_to(RAIZ))
    except ValueError:
        return str(ruta)


def _guardar(registro: dict, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{registro['frame_id']}.json"
    ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    r = registro["resumen"]
    conteo = ", ".join(f"{c}={n}" for c, n in r["por_clase"].items() if n)
    print(f"  {registro['frame_id']}: {r['total']} detecciones ({conteo or 'ninguna'})")
    return ruta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exporta detecciones YOLO a JSON por frame.")
    parser.add_argument("--fuente", required=True, help="Imagen o carpeta de imágenes.")
    parser.add_argument("--pesos", help="Ruta a best.pt (por defecto YOLO_WEIGHTS o runs_nju/).")
    parser.add_argument("--modo", choices=["auto", "frames", "tiles"], default="auto")
    parser.add_argument("--secuencia", help="ID de secuencia a guardar en los metadatos.")
    parser.add_argument("--destino", help="Carpeta de salida (por defecto DETECCIONES_DIR).")
    args = parser.parse_args(argv)

    config = cargar_config()
    from ultralytics import YOLO

    pesos = encontrar_pesos(config, args.pesos)
    device = elegir_dispositivo()
    imagenes = listar_imagenes(Path(args.fuente).expanduser())
    modo = detectar_modo(imagenes, config) if args.modo == "auto" else args.modo
    destino = Path(args.destino).expanduser() if args.destino else config.detecciones_dir

    print(f"Modelo: {pesos} | dispositivo: {device} | modo: {modo} | imágenes: {len(imagenes)}")
    model = YOLO(str(pesos))
    modelo_info = {
        "pesos": _relativa(Path(pesos)),
        "conf": config.yolo_conf,
        "imgsz": config.yolo_imgsz,
        "tile_size": config.tile_size,
        "overlap": config.tile_overlap,
        "dedup_iou": config.dedup_iou,
    }

    exportar = exportar_tiles if modo == "tiles" else exportar_frames
    rutas = exportar(model, imagenes, config, device, modelo_info, args.secuencia, destino)
    print(f"{len(rutas)} archivos JSON guardados en {destino}")
    print("Siguiente paso: python -m agente.indexar")


if __name__ == "__main__":
    main()
