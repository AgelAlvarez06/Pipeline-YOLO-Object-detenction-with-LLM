"""
Agente de análisis RAG sobre los resultados del pipeline YOLO (NJU-HARD).

Completa el pipeline:  YOLO -> JSON por frame -> documentos -> ChromaDB -> agente LLM.

Uso:
    python agente.py exportar --fuente nju_yolo/images/val   # 1. detecciones a JSON
    python agente.py indexar [--reset]                        # 2. RAG en ChromaDB
    python agente.py chat                                     # 3. conversar
    python agente.py chat -p "¿Cuántos peatones hay?"         #    una sola pregunta

Toda la configuración se lee de .env (ver .env.example).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

# Evita la telemetría de Chroma y el aviso de paralelismo de tokenizers.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain.tools import tool  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # los frames de NJU-HARD superan el límite de Pillow

RAIZ = Path(__file__).resolve().parent

# Mismas clases que yolo.py (no se importa yolo.py porque exige GPU NVIDIA al cargar).
CLASS_NAMES = {0: "pedestrian", 1: "car", 2: "bus", 3: "truck"}
NOMBRES_ES = {"pedestrian": "peatones", "car": "autos", "bus": "autobuses", "truck": "camiones"}


# =============================================================================
# 1. Configuración (.env)
# =============================================================================
# Proveedor -> variable con la API key (None = no necesita clave)
PROVEEDORES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",  # Gemini con API key de Google AI Studio
    "ollama": None,
}
ALIAS_PROVEEDOR = {"gemini": "google_genai", "google": "google_genai"}
NIVELES_EFFORT = {"low", "medium", "high", "xhigh", "max"}


class ErrorConfiguracion(RuntimeError):
    """Configuración inválida o incompleta en .env."""


def _leer(nombre: str, defecto: str = "") -> str:
    valor = os.getenv(nombre)
    return defecto if valor is None or not valor.strip() else valor.strip()


def _numero(nombre: str, defecto: str, tipo):
    valor = _leer(nombre, defecto)
    try:
        return tipo(valor)
    except ValueError as exc:
        raise ErrorConfiguracion(f"{nombre}={valor!r} no es un {tipo.__name__} válido.") from exc


def _ruta(valor: str) -> Path:
    ruta = Path(valor).expanduser()
    return ruta if ruta.is_absolute() else (RAIZ / ruta).resolve()


@dataclass(frozen=True)
class Config:
    llm_provider: str
    llm_model: str
    llm_temperature: float | None
    llm_max_tokens: int
    llm_effort: str | None
    llm_timeout: float
    llm_max_retries: int
    ollama_base_url: str
    embeddings_model: str
    embeddings_device: str
    chroma_dir: Path
    chroma_collection: str
    rag_top_k: int
    yolo_weights: Path | None
    yolo_conf: float
    yolo_imgsz: int
    tile_size: int
    tile_overlap: int
    dedup_iou: float
    grid_filas: int
    grid_columnas: int
    detecciones_dir: Path
    metricas_path: Path
    graficas_dir: Path
    runs_dir: Path

    def validar_llm(self) -> None:
        """Comprueba que el proveedor elegido tenga lo necesario para arrancar."""
        if self.llm_provider not in PROVEEDORES:
            raise ErrorConfiguracion(
                f"LLM_PROVIDER={self.llm_provider!r} no soportado. "
                f"Opciones: anthropic, openai, gemini, ollama."
            )
        if not self.llm_model:
            raise ErrorConfiguracion("Falta LLM_MODEL en .env.")
        variable = PROVEEDORES[self.llm_provider]
        if variable and not _leer(variable):
            raise ErrorConfiguracion(
                f"Falta {variable} en .env (necesaria para LLM_PROVIDER="
                f"{self.llm_provider}). Añádela o cambia de proveedor."
            )
        if self.llm_effort and self.llm_effort not in NIVELES_EFFORT:
            raise ErrorConfiguracion(
                f"LLM_EFFORT={self.llm_effort!r} no válido. "
                f"Opciones: {', '.join(sorted(NIVELES_EFFORT))}."
            )


def cargar_config(env_path: Path | None = None) -> Config:
    """Lee .env (sin sobrescribir variables ya exportadas) y construye la Config."""
    load_dotenv(env_path or RAIZ / ".env", override=False)
    temperatura = _leer("LLM_TEMPERATURE")
    pesos = _leer("YOLO_WEIGHTS")
    proveedor = _leer("LLM_PROVIDER", "anthropic").lower()

    config = Config(
        llm_provider=ALIAS_PROVEEDOR.get(proveedor, proveedor),
        llm_model=_leer("LLM_MODEL", "claude-opus-5-5"),
        llm_temperature=float(temperatura) if temperatura else None,
        llm_max_tokens=_numero("LLM_MAX_TOKENS", "16000", int),
        llm_effort=_leer("LLM_EFFORT").lower() or None,
        llm_timeout=_numero("LLM_TIMEOUT", "120", float),
        llm_max_retries=_numero("LLM_MAX_RETRIES", "2", int),
        ollama_base_url=_leer("OLLAMA_BASE_URL", "http://localhost:11434"),
        embeddings_model=_leer(
            "EMBEDDINGS_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        embeddings_device=_leer("EMBEDDINGS_DEVICE", "cpu"),
        chroma_dir=_ruta(_leer("CHROMA_DIR", "data/chroma")),
        chroma_collection=_leer("CHROMA_COLLECTION", "nju_resultados"),
        rag_top_k=_numero("RAG_TOP_K", "6", int),
        yolo_weights=_ruta(pesos) if pesos else None,
        yolo_conf=_numero("YOLO_CONF", "0.25", float),
        yolo_imgsz=_numero("YOLO_IMGSZ", "640", int),
        tile_size=_numero("TILE_SIZE", "1024", int),
        tile_overlap=_numero("TILE_OVERLAP", "128", int),
        dedup_iou=_numero("DEDUP_IOU", "0.6", float),
        grid_filas=_numero("GRID_FILAS", "4", int),
        grid_columnas=_numero("GRID_COLUMNAS", "4", int),
        detecciones_dir=_ruta(_leer("DETECCIONES_DIR", "resultados/detecciones")),
        metricas_path=_ruta(_leer("METRICAS_PATH", "metricas.json")),
        graficas_dir=_ruta(_leer("GRAFICAS_DIR", "resultados/graficas")),
        runs_dir=RAIZ / "runs_nju",
    )
    if config.tile_overlap >= config.tile_size:
        raise ErrorConfiguracion("TILE_OVERLAP debe ser menor que TILE_SIZE.")
    if config.grid_filas < 1 or config.grid_columnas < 1:
        raise ErrorConfiguracion("GRID_FILAS y GRID_COLUMNAS deben ser >= 1.")
    return config


# =============================================================================
# 2. LLM configurable (anthropic | openai | gemini | ollama)
# =============================================================================
def crear_llm(config: Config):
    from langchain.chat_models import init_chat_model

    config.validar_llm()
    proveedor = config.llm_provider
    kwargs: dict = {}
    if proveedor != "ollama":
        # Sin esto, un fallo del servidor (p. ej. 504 de Gemini) se reintenta varias
        # veces sin límite de tiempo y el chat parece colgado durante minutos.
        kwargs["timeout"] = config.llm_timeout
        kwargs["max_retries"] = config.llm_max_retries

    if proveedor == "anthropic":
        # Los modelos Claude actuales rechazan temperature (400): no se envía.
        # La profundidad de razonamiento se controla con effort.
        kwargs["max_tokens"] = config.llm_max_tokens
        if config.llm_effort:
            kwargs["output_config"] = {"effort": config.llm_effort}
    elif proveedor == "openai":
        kwargs["max_tokens"] = config.llm_max_tokens
        if config.llm_temperature is not None:
            kwargs["temperature"] = config.llm_temperature
    elif proveedor == "google_genai":
        # Gemini (AI Studio); la clave se lee de GOOGLE_API_KEY. No se envía temperature:
        # Google recomienda el valor por defecto (1.0) en Gemini 3; bajarlo puede causar bucles.
        kwargs["max_output_tokens"] = config.llm_max_tokens
        if config.llm_effort:
            # Gemini 3 acepta minimal | low | medium | high
            kwargs["thinking_level"] = {"xhigh": "high", "max": "high"}.get(
                config.llm_effort, config.llm_effort
            )
    elif proveedor == "ollama":
        kwargs["base_url"] = config.ollama_base_url
        kwargs["num_predict"] = config.llm_max_tokens
        if config.llm_temperature is not None:
            kwargs["temperature"] = config.llm_temperature

    return init_chat_model(config.llm_model, model_provider=proveedor, **kwargs)


def describir_llm(config: Config) -> str:
    con_effort = config.llm_provider in {"anthropic", "google_genai"} and config.llm_effort
    extra = f", effort={config.llm_effort}" if con_effort else ""
    return f"{config.llm_provider} / {config.llm_model}{extra}"


# =============================================================================
# 3. Exportación: YOLO -> un JSON por frame
# =============================================================================
# Modos:
# - frames: cada imagen es un frame completo. Si supera TILE_SIZE se corta en tiles
#   solapados, se predice cada tile, las cajas pasan a coordenadas del frame y se
#   deduplican con NMS por clase (DEDUP_IOU).
# - tiles: imágenes de yolo.py ({frame_id}_{fila}_{columna}.jpg), reagrupadas por frame.
#   yolo.py solo guarda tiles con etiquetas y no guarda el tamaño del frame, así que la
#   posición global es aproximada y la cobertura parcial.
# - auto: "tiles" si todas siguen ese patrón y no superan TILE_SIZE; si no, "frames".
EXTENSIONES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PATRON_TILE = re.compile(r"^(?P<frame>.+)_(?P<fila>\d+)_(?P<col>\d+)$")
LOTE = 8  # tiles por llamada a model.predict


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
    for i, det in enumerate(sorted(detecciones, key=lambda d: -d["confianza"])):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        celda = celda_de((x1 + x2) / 2, (y1 + y2) / 2, ancho, alto, filas, cols)
        por_clase[det["clase"]] += 1
        suma_conf[det["clase"]] += det["confianza"]
        por_celda[celda][det["clase"]] += 1
        salida.append({
            "id": i,
            "clase": det["clase"],
            "clase_id": det["clase_id"],
            "confianza": round(det["confianza"], 4),
            "bbox_xyxy": [round(v, 1) for v in det["bbox_xyxy"]],
            "celda": celda,
        })

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
            "conteo_por_celda": {c: dict(n) for c, n in sorted(por_celda.items())},
        },
        "detecciones": salida,
    }


def encontrar_pesos(config: Config, pesos: str | None) -> Path:
    if pesos:
        ruta = Path(pesos).expanduser()
        return ruta if ruta.is_absolute() or not (RAIZ / ruta).exists() else RAIZ / ruta
    if config.yolo_weights:
        if not config.yolo_weights.exists():
            raise FileNotFoundError(f"YOLO_WEIGHTS no existe: {config.yolo_weights}")
        return config.yolo_weights
    candidatos = sorted(config.runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
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
    cajas = []
    for (x1, y1, x2, y2), conf, cls in zip(
        resultado.boxes.xyxy.tolist(), resultado.boxes.conf.tolist(), resultado.boxes.cls.tolist()
    ):
        cajas.append({
            "clase_id": int(cls),
            "clase": resultado.names[int(cls)],
            "confianza": float(conf),
            "bbox_xyxy": [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
        })
    return cajas


def deduplicar(cajas: list[dict], iou: float) -> list[dict]:
    """NMS por clase para eliminar cajas repetidas en zonas de solapamiento."""
    if not cajas:
        return []
    import torch
    from torchvision.ops import batched_nms

    keep = batched_nms(
        torch.tensor([c["bbox_xyxy"] for c in cajas], dtype=torch.float32),
        torch.tensor([c["confianza"] for c in cajas], dtype=torch.float32),
        torch.tensor([c["clase_id"] for c in cajas], dtype=torch.int64),
        iou,
    ).tolist()
    return [cajas[i] for i in keep]


def _predecir(model, imagenes, config: Config, device) -> list:
    return model.predict(
        source=imagenes, imgsz=config.yolo_imgsz, conf=config.yolo_conf, device=device, verbose=False
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


def _relativa(ruta: Path) -> str:
    try:
        return str(ruta.resolve().relative_to(RAIZ))
    except ValueError:
        return str(ruta)


def _guardar_registro(registro: dict, destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{registro['frame_id']}.json"
    ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    r = registro["resumen"]
    conteo = ", ".join(f"{c}={n}" for c, n in r["por_clase"].items() if n)
    print(f"  {registro['frame_id']}: {r['total']} detecciones ({conteo or 'ninguna'})")
    return ruta


def exportar(
    config: Config,
    fuente: Path,
    pesos: str | None = None,
    modo: str = "auto",
    secuencia: str | None = None,
    destino: Path | None = None,
) -> list[Path]:
    from ultralytics import YOLO

    ruta_pesos = encontrar_pesos(config, pesos)
    device = elegir_dispositivo()
    imagenes = listar_imagenes(fuente)
    modo = detectar_modo(imagenes, config) if modo == "auto" else modo
    destino = destino or config.detecciones_dir
    print(f"Modelo: {ruta_pesos} | dispositivo: {device} | modo: {modo} | imágenes: {len(imagenes)}")

    model = YOLO(str(ruta_pesos))
    modelo_info = {
        "pesos": _relativa(Path(ruta_pesos)),
        "conf": config.yolo_conf,
        "imgsz": config.yolo_imgsz,
        "tile_size": config.tile_size,
        "overlap": config.tile_overlap,
        "dedup_iou": config.dedup_iou,
    }
    comunes = dict(secuencia=secuencia, modelo=modelo_info, config=config)
    rutas = []

    if modo == "frames":
        for ruta in imagenes:
            with Image.open(ruta) as img:
                imagen = img.convert("RGB")
            cajas = predecir_frame(model, imagen, config, device)
            ancho, alto = imagen.size
            imagen.close()
            n_tiles = len(tile_starts(ancho, config.tile_size, config.tile_overlap)) * len(
                tile_starts(alto, config.tile_size, config.tile_overlap)
            )
            registro = construir_registro(
                frame_id=ruta.stem, detecciones=cajas, ancho=ancho, alto=alto,
                ruta_imagen=_relativa(ruta), split=inferir_split(ruta), n_tiles=n_tiles, **comunes,
            )
            rutas.append(_guardar_registro(registro, destino))
        return rutas

    # modo tiles: reagrupar por frame con posiciones aproximadas
    paso = config.tile_size - config.tile_overlap
    grupos: dict[str, list[tuple[Path, int, int]]] = defaultdict(list)
    for ruta in imagenes:
        m = PATRON_TILE.match(ruta.stem)
        grupos[m["frame"]].append((ruta, int(m["fila"]), int(m["col"])))

    for frame_id, tiles in sorted(grupos.items()):
        cajas: list[dict] = []
        ancho = alto = 0
        for i in range(0, len(tiles), LOTE):
            lote = tiles[i : i + LOTE]
            for (ruta, fila, col), resultado in zip(
                lote, _predecir(model, [str(t[0]) for t in lote], config, device)
            ):
                dx, dy = col * paso, fila * paso
                alto_t, ancho_t = resultado.orig_shape
                ancho, alto = max(ancho, dx + ancho_t), max(alto, dy + alto_t)
                cajas.extend(_cajas(resultado, dx, dy))
        registro = construir_registro(
            frame_id=frame_id, detecciones=deduplicar(cajas, config.dedup_iou),
            ancho=ancho, alto=alto, ruta_imagen=_relativa(tiles[0][0].parent),
            split=inferir_split(tiles[0][0]), coordenadas="aproximadas",
            cobertura="parcial (solo tiles con etiquetas guardados por yolo.py)",
            n_tiles=len(tiles), **comunes,
        )
        rutas.append(_guardar_registro(registro, destino))
    return rutas


# =============================================================================
# 4. Documentos para el RAG
# =============================================================================
# Las cifras se precalculan aquí y se escriben en el texto, para que el LLM las cite
# en vez de contar cajas. Cada documento lleva un `fuente` estable que sirve de cita.
TIPOS = ("resumen_global", "resumen_frame", "region_frame", "metricas_modelo")


def nombre_es(clase: str) -> str:
    return NOMBRES_ES.get(clase, clase)


def _tercio(indice: int, total: int, nombres: tuple[str, str, str]) -> str:
    posicion = (indice + 0.5) / total
    return nombres[0] if posicion < 1 / 3 else nombres[1] if posicion < 2 / 3 else nombres[2]


def _fila_col(celda: str) -> tuple[int, int]:
    return int(celda[1 : celda.index("c")]), int(celda[celda.index("c") + 1 :])


def describir_celda(celda: str, filas: int, columnas: int) -> str:
    """'r0c3' -> 'fila 1 de 4, columna 4 de 4 (zona superior-derecha)'."""
    fila, col = _fila_col(celda)
    vertical = _tercio(fila, filas, ("superior", "centro", "inferior"))
    horizontal = _tercio(col, columnas, ("izquierda", "centro", "derecha"))
    zona = "central" if vertical == horizontal == "centro" else f"{vertical}-{horizontal}"
    return f"fila {fila + 1} de {filas}, columna {col + 1} de {columnas} (zona {zona})"


def _lista_conteos(conteo: dict[str, int]) -> str:
    partes = [f"{n} {nombre_es(c)}" for c, n in conteo.items() if n]
    return ", ".join(partes) if partes else "ninguna detección"


def _doc(texto: str, id_: str, **meta) -> Document:
    # Chroma solo admite str/int/float/bool en metadatos: se descartan los None.
    metadatos = {k: v for k, v in {"fuente": id_, **meta}.items() if v is not None}
    return Document(page_content=texto, id=id_, metadata=metadatos)


def _nota_cobertura(registro: dict) -> str:
    cobertura = registro.get("cobertura") or {}
    tipo = cobertura.get("tipo", "completa")
    if tipo == "completa":
        return ""
    return (
        f" Atención: cobertura {tipo}; coordenadas {cobertura.get('coordenadas', '?')}"
        f" ({cobertura.get('n_tiles', '?')} tiles analizados), así que las zonas y conteos"
        " no representan el frame completo."
    )


def documentos_frame(registro: dict) -> list[Document]:
    frame_id = registro["frame_id"]
    resumen = registro["resumen"]
    grid = registro.get("grid", {"filas": 4, "columnas": 4})
    filas, cols = grid["filas"], grid["columnas"]
    modelo = registro.get("modelo", {})
    base = dict(frame_id=frame_id, secuencia=registro.get("secuencia"), split=registro.get("split"))
    nota = _nota_cobertura(registro)

    confs = ", ".join(
        f"{nombre_es(c)} {v:.2f}" for c, v in resumen.get("confianza_media_por_clase", {}).items()
    ) or "n/d"
    celdas = resumen.get("conteo_por_celda", {})
    top = sorted(celdas.items(), key=lambda kv: -sum(kv[1].values()))[:3]
    top_txt = "; ".join(
        f"{describir_celda(c, filas, cols)}: {sum(v.values())} ({_lista_conteos(v)})" for c, v in top
    ) or "ninguna"
    imagen = registro.get("imagen", {})
    texto = (
        f"Resumen del frame {frame_id} (secuencia {registro.get('secuencia') or 'desconocida'},"
        f" split {registro.get('split', 'desconocido')}, imagen de {imagen.get('ancho')}x"
        f"{imagen.get('alto')} px). El modelo YOLO detectó {resumen['total']} objetos en total:"
        f" {_lista_conteos(resumen['por_clase'])}. Confianza media por clase: {confs}."
        f" Zonas con más detecciones: {top_txt}."
        f" Parámetros: confianza mínima {modelo.get('conf')}, imgsz {modelo.get('imgsz')},"
        f" tiles de {modelo.get('tile_size')} px con solapamiento {modelo.get('overlap')},"
        f" deduplicación IoU {modelo.get('dedup_iou')}, pesos {modelo.get('pesos')}." + nota
    )
    docs = [_doc(texto, f"resumen_frame/{frame_id}", tipo="resumen_frame", total=resumen["total"], **base)]

    confs_celda: dict[str, dict[str, list[float]]] = {}
    for det in registro.get("detecciones", []):
        confs_celda.setdefault(det["celda"], {}).setdefault(det["clase"], []).append(det["confianza"])

    for celda, conteo in celdas.items():
        detalle = []
        for clase, n in conteo.items():
            valores = confs_celda.get(celda, {}).get(clase, [])
            media = f" (confianza media {sum(valores) / len(valores):.2f})" if valores else ""
            detalle.append(f"{n} {nombre_es(clase)}{media}")
        total = sum(conteo.values())
        texto = (
            f"En el frame {frame_id}, la celda {celda} ({describir_celda(celda, filas, cols)})"
            f" tiene {total} detecciones: {', '.join(detalle)}."
            f" Representa el {100 * total / max(1, resumen['total']):.1f}% de las detecciones"
            " del frame." + nota
        )
        docs.append(_doc(
            texto, f"region_frame/{frame_id}/{celda}", tipo="region_frame", celda=celda, total=total, **base
        ))
    return docs


def documento_global(registros: list[dict]) -> Document:
    totales: Counter = Counter()
    for r in registros:
        totales.update(r["resumen"]["por_clase"])

    def vehiculos(r):
        return sum(r["resumen"]["por_clase"].get(c, 0) for c in ("car", "bus", "truck"))

    ranking = sorted(registros, key=lambda r: -r["resumen"]["total"])
    mas_vehiculos = max(registros, key=vehiculos)
    total = sum(totales.values())
    reparto = ", ".join(f"{nombre_es(c)} {n} ({100 * n / max(1, total):.1f}%)" for c, n in totales.items())
    top = "; ".join(f"{r['frame_id']} ({r['resumen']['total']})" for r in ranking[:5])
    texto = (
        f"Resumen global de {len(registros)} frames analizados: {total} detecciones en total."
        f" Distribución por clase: {reparto}."
        f" Frames con más detecciones: {top}."
        + (
            f" Frame con más vehículos (autos+autobuses+camiones): {mas_vehiculos['frame_id']}"
            f" con {vehiculos(mas_vehiculos)}."
            if vehiculos(mas_vehiculos)
            else " El modelo no detectó ningún vehículo (auto, autobús ni camión) en ningún frame."
        )
        + f" Frames indexados: {', '.join(r['frame_id'] for r in registros)}."
    )
    return _doc(texto, "resumen_global", tipo="resumen_global", n_frames=len(registros))


def _interpretar_clase(clase: str, m: dict) -> str:
    tp, fp, fn = m["TP"], m["FP"], m["FN"]
    es = nombre_es(clase)
    if tp == fp == fn == 0:
        return f"No hubo {es} reales ni predichos; las métricas de esta clase no aplican."
    if tp == 0 and fn == 0:
        return (
            f"No había {es} en el ground truth, pero el modelo predijo {fp}: todas son falsos"
            " positivos. La precisión es 0 y el recall está indefinido (no hay objetos reales;"
            " el 0.0 del archivo no significa que el modelo haya omitido objetos)."
        )
    if tp == 0 and fp == 0:
        return f"El modelo no predijo ningún objeto de la clase {es} y omitió los {fn} reales."
    partes = []
    if fp > tp:
        partes.append(f"genera más falsos positivos ({fp}) que aciertos ({tp}), sobreestima")
    if fn > tp:
        partes.append(f"omite más objetos reales ({fn}) de los que encuentra ({tp})")
    return "El modelo " + " y ".join(partes) + "." if partes else "Desempeño equilibrado."


def documentos_metricas(metricas: dict) -> list[Document]:
    dedup = metricas.get("cajas_predichas_deduplicacion", {})
    dedup_txt = ", ".join(
        f"{nombre_es(c)} {v['antes']}→{v['despues']}" for c, v in dedup.items() if v.get("antes")
    )
    contexto = (
        f"Evaluación sobre {metricas.get('frames_evaluados', '?')} frame(s) completo(s),"
        f" deduplicación entre tiles con IoU {metricas.get('dedup_iou', '?')}"
        + (f" (cajas antes→después: {dedup_txt})" if dedup_txt else "") + "."
    )
    docs = []
    for umbral, datos in metricas.get("por_umbral", {}).items():
        mi, ma = datos["micro"], datos["macro"]
        texto = (
            f"Métricas globales del modelo YOLO con umbral {umbral}. {contexto}"
            f" Micro: TP {mi['TP']}, FP {mi['FP']}, FN {mi['FN']}, precisión {mi['precision']:.4f},"
            f" recall {mi['recall']:.4f}, F1 {mi['F1']:.4f}, IoU medio de aciertos"
            f" {mi.get('IoU_medio_TP')}. Macro: precisión {ma['precision']:.4f},"
            f" recall {ma['recall']:.4f}, F1 {ma['F1']:.4f}."
            f" Interpretación: de cada 100 detecciones, unas {100 * mi['precision']:.0f} son"
            f" correctas, y se encuentra el {100 * mi['recall']:.0f}% de los objetos reales."
        )
        docs.append(_doc(texto, f"metricas_modelo/{umbral}", tipo="metricas_modelo", umbral_iou=umbral))
        for clase, m in datos.get("por_clase", {}).items():
            texto = (
                f"Métricas de la clase {clase} ({nombre_es(clase)}) con umbral {umbral}:"
                f" TP {m['TP']}, FP {m['FP']}, FN {m['FN']}, precisión {m['precision']:.4f},"
                f" recall {m['recall']:.4f}, F1 {m['F1']:.4f}, IoU medio de aciertos"
                f" {m.get('IoU_medio_TP')}. {_interpretar_clase(clase, m)} {contexto}"
            )
            docs.append(_doc(
                texto, f"metricas_modelo/{umbral}/{clase}", tipo="metricas_modelo",
                umbral_iou=umbral, clase=clase,
            ))
    return docs


def cargar_registros(detecciones_dir: Path) -> list[dict]:
    if not detecciones_dir.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(detecciones_dir.glob("*.json"))]


def construir_documentos(detecciones_dir: Path, metricas_path: Path) -> list[Document]:
    registros = cargar_registros(detecciones_dir)
    docs: list[Document] = []
    if registros:
        docs.append(documento_global(registros))
        for registro in registros:
            docs.extend(documentos_frame(registro))
    if metricas_path.exists():
        docs.extend(documentos_metricas(json.loads(metricas_path.read_text(encoding="utf-8"))))
    return docs


# =============================================================================
# 5. Indexación en ChromaDB (embeddings locales)
# =============================================================================
@lru_cache(maxsize=2)
def crear_embeddings(modelo: str, dispositivo: str):
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    if (dispositivo == "mps" and not torch.backends.mps.is_available()) or (
        dispositivo == "cuda" and not torch.cuda.is_available()
    ):
        dispositivo = "cpu"
    return HuggingFaceEmbeddings(
        model_name=modelo,
        model_kwargs={"device": dispositivo},
        encode_kwargs={"normalize_embeddings": True},
    )


def abrir_vectorstore(config: Config):
    from langchain_chroma import Chroma

    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.chroma_collection,
        embedding_function=crear_embeddings(config.embeddings_model, config.embeddings_device),
        persist_directory=str(config.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def indexar(config: Config, reset: bool = False) -> Counter:
    """Upsert idempotente por id; elimina documentos de frames que ya no existen."""
    docs = construir_documentos(config.detecciones_dir, config.metricas_path)
    if not docs:
        raise RuntimeError(
            f"No hay nada que indexar: ni JSON en {config.detecciones_dir} ni {config.metricas_path}."
            " Ejecuta antes: python agente.py exportar --fuente <imágenes>"
        )
    store = abrir_vectorstore(config)
    if reset:
        store.delete_collection()
        store = abrir_vectorstore(config)

    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("IDs de documentos duplicados; revisa los frame_id de los JSON.")
    obsoletos = sorted(set(store.get(include=[])["ids"]) - set(ids))
    if obsoletos:
        store.delete(ids=obsoletos)
    store.add_documents(docs, ids=ids)  # Chroma hace upsert por id

    por_tipo = Counter(d.metadata["tipo"] for d in docs)
    por_tipo["_obsoletos_eliminados"] = len(obsoletos)
    return por_tipo


# =============================================================================
# 6. Herramientas del agente
# =============================================================================
# - buscar_resultados: búsqueda semántica en ChromaDB (RAG).
# - generar_grafica: lee los JSON / metricas.json directamente (no el RAG), así los
#   números de la gráfica son exactos.
TipoGrafica = Literal["barras_clases", "histograma_confianza", "heatmap_celdas", "metricas_por_clase"]


class ErrorHerramienta(ValueError):
    """Error con mensaje legible que se devuelve al LLM."""


class Herramientas:
    """Agrupa las herramientas y el estado compartido (vectorstore, últimas fuentes)."""

    def __init__(self, config: Config, vectorstore=None):
        self.config = config
        self._vectorstore = vectorstore
        self.ultimas_fuentes: list[dict] = []

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = abrir_vectorstore(self.config)
        return self._vectorstore

    # ------------------------------------------------------------------ RAG
    def buscar(self, consulta: str, tipo: str | None = None, frame_id: str | None = None) -> str:
        if tipo and tipo not in TIPOS:
            raise ErrorHerramienta(f"tipo '{tipo}' no válido. Opciones: {', '.join(TIPOS)}.")
        condiciones = [{k: v} for k, v in (("tipo", tipo), ("frame_id", frame_id)) if v]
        filtro = condiciones[0] if len(condiciones) == 1 else {"$and": condiciones} if condiciones else None

        resultados = self.vectorstore.similarity_search_with_score(
            consulta, k=self.config.rag_top_k, filter=filtro
        )
        self.ultimas_fuentes.extend(
            {"fuente": d.metadata.get("fuente"), "distancia": round(s, 4), "texto": d.page_content}
            for d, s in resultados
        )
        if not resultados:
            ayuda = " Prueba sin filtros." if filtro else " ¿Está indexado? (python agente.py indexar)"
            return "No se encontraron documentos para esa consulta." + ayuda
        return "\n\n".join(f"[fuente: {d.metadata.get('fuente')}]\n{d.page_content}" for d, _ in resultados)

    # ------------------------------------------------------------- Gráficas
    def _registros(self) -> list[dict]:
        registros = cargar_registros(self.config.detecciones_dir)
        if not registros:
            raise ErrorHerramienta(
                f"No hay detecciones en {self.config.detecciones_dir}. "
                "Hay que ejecutar antes: python agente.py exportar"
            )
        return registros

    def _registro(self, frame_id: str | None) -> dict:
        registros = self._registros()
        if frame_id is None:
            return max(registros, key=lambda r: r["resumen"]["total"])
        for r in registros:
            if r["frame_id"] == frame_id:
                return r
        disponibles = ", ".join(r["frame_id"] for r in registros[:20])
        raise ErrorHerramienta(f"frame_id '{frame_id}' no encontrado; disponibles: {disponibles}.")

    def graficar(self, tipo_grafica: str, frame_id: str | None = None, clase: str | None = None,
                 umbral_iou: str = "IoU_0.5") -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        funciones = {
            "barras_clases": self._barras_clases,
            "histograma_confianza": self._histograma_confianza,
            "heatmap_celdas": self._heatmap_celdas,
            "metricas_por_clase": self._metricas_por_clase,
        }
        if tipo_grafica not in funciones:
            raise ErrorHerramienta(f"tipo_grafica no válido. Opciones: {', '.join(funciones)}.")

        self.config.graficas_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        try:
            nombre, resumen = funciones[tipo_grafica](ax, frame_id, clase, umbral_iou)
            fig.tight_layout()
            ruta = self.config.graficas_dir / f"{nombre}.png"
            fig.savefig(ruta)
        finally:
            plt.close(fig)
        return f"Gráfica guardada en: {ruta}\nDatos graficados: {resumen}"

    def _barras_clases(self, ax, frame_id, clase, umbral):
        if frame_id:
            conteo = self._registro(frame_id)["resumen"]["por_clase"]
            titulo = f"Detecciones por clase, frame {frame_id}"
        else:
            conteo = Counter()
            for r in self._registros():
                conteo.update(r["resumen"]["por_clase"])
            conteo, titulo = dict(conteo), "Detecciones por clase (todos los frames)"
        barras = ax.bar([nombre_es(c) for c in conteo], list(conteo.values()), color="#2a78d6")
        ax.bar_label(barras)
        ax.set(title=titulo, xlabel="Clase", ylabel="Número de detecciones")
        return f"barras_clases_{frame_id or 'global'}", json.dumps(conteo, ensure_ascii=False)

    def _histograma_confianza(self, ax, frame_id, clase, umbral):
        registros = [self._registro(frame_id)] if frame_id else self._registros()
        valores = [
            d["confianza"] for r in registros for d in r["detecciones"] if clase is None or d["clase"] == clase
        ]
        if not valores:
            raise ErrorHerramienta(f"No hay detecciones de '{clase}' para graficar.")
        ax.hist(valores, bins=20, range=(0, 1), color="#2a78d6", edgecolor="white")
        ax.axvline(self.config.yolo_conf, color="#d03b3b", linestyle="--", label="umbral conf")
        ax.legend()
        sujeto = nombre_es(clase) if clase else "todas las clases"
        ax.set(title=f"Confianza de las detecciones ({sujeto}, {frame_id or 'todos los frames'})",
               xlabel="Confianza", ylabel="Número de detecciones")
        media = sum(valores) / len(valores)
        resumen = f"n={len(valores)}, media={media:.3f}, min={min(valores):.3f}, max={max(valores):.3f}"
        return f"histograma_confianza_{frame_id or 'global'}_{clase or 'todas'}", resumen

    def _heatmap_celdas(self, ax, frame_id, clase, umbral):
        r = self._registro(frame_id)
        filas, cols = r["grid"]["filas"], r["grid"]["columnas"]
        matriz = [[0] * cols for _ in range(filas)]
        for celda, conteo in r["resumen"]["conteo_por_celda"].items():
            f, c = _fila_col(celda)
            matriz[f][c] = conteo.get(clase, 0) if clase else sum(conteo.values())
        im = ax.imshow(matriz, cmap="viridis")
        for f in range(filas):
            for c in range(cols):
                ax.text(c, f, matriz[f][c], ha="center", va="center", color="white", fontsize=9)
        ax.figure.colorbar(im, ax=ax, label="Detecciones")
        ax.set_xticks(range(cols), [f"col {i + 1}" for i in range(cols)])
        ax.set_yticks(range(filas), [f"fila {i + 1}" for i in range(filas)])
        sujeto = nombre_es(clase) if clase else "todas las clases"
        ax.set_title(f"Distribución espacial ({sujeto}), frame {r['frame_id']}")
        return f"heatmap_{r['frame_id']}_{clase or 'todas'}", f"frame={r['frame_id']}, matriz={matriz}"

    def _metricas_por_clase(self, ax, frame_id, clase, umbral):
        if not self.config.metricas_path.exists():
            raise ErrorHerramienta(f"No existe {self.config.metricas_path}.")
        por_umbral = json.loads(self.config.metricas_path.read_text(encoding="utf-8")).get("por_umbral", {})
        if umbral not in por_umbral:
            raise ErrorHerramienta(f"umbral_iou '{umbral}' no existe. Opciones: {', '.join(por_umbral)}.")
        por_clase = por_umbral[umbral]["por_clase"]
        clases = list(por_clase)
        ancho = 0.27
        for i, (metrica, color) in enumerate((("precision", "#2a78d6"), ("recall", "#e08a1e"), ("F1", "#3a9b5c"))):
            xs = [j + (i - 1) * ancho for j in range(len(clases))]
            ax.bar(xs, [por_clase[c][metrica] for c in clases], ancho, label=metrica, color=color)
        ax.set_xticks(range(len(clases)), [nombre_es(c) for c in clases])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set(title=f"Métricas por clase ({umbral})", xlabel="Clase", ylabel="Valor")
        resumen = {c: {k: por_clase[c][k] for k in ("TP", "FP", "FN", "precision", "recall", "F1")} for c in clases}
        return f"metricas_por_clase_{umbral}", json.dumps(resumen, ensure_ascii=False)

    # --------------------------------------------------- Tools para LangChain
    def como_tools(self) -> list:
        herramientas = self

        @tool
        def buscar_resultados(consulta: str, tipo: str | None = None, frame_id: str | None = None) -> str:
            """Busca en la base de conocimiento los resultados del detector YOLO.

            Úsala SIEMPRE antes de responder con cifras. Devuelve fragmentos con su
            fuente entre corchetes, que debes citar.

            Args:
                consulta: pregunta o tema en español (p. ej. "peatones por zona").
                tipo: filtro opcional. "resumen_global" (totales de todos los frames),
                    "resumen_frame" (totales de un frame), "region_frame" (conteos por
                    celda de la cuadrícula, para preguntas espaciales) o "metricas_modelo"
                    (precisión, recall, F1, TP/FP/FN del modelo).
                frame_id: filtro opcional para un frame concreto.
            """
            try:
                return herramientas.buscar(consulta, tipo, frame_id)
            except ErrorHerramienta as exc:
                return f"Error: {exc}"

        @tool
        def generar_grafica(
            tipo_grafica: TipoGrafica,
            frame_id: str | None = None,
            clase: Literal["pedestrian", "car", "bus", "truck"] | None = None,
            umbral_iou: Literal["IoU_0.5", "IoU_0.75"] = "IoU_0.5",
        ) -> str:
            """Genera una gráfica PNG con datos exactos y devuelve su ruta y los valores.

            Args:
                tipo_grafica: "barras_clases" (conteo por clase), "histograma_confianza"
                    (distribución de confianza), "heatmap_celdas" (dónde se concentran las
                    detecciones en la cuadrícula del frame) o "metricas_por_clase"
                    (precisión/recall/F1 de metricas.json).
                frame_id: frame concreto. Si se omite: barras e histograma usan todos los
                    frames y el heatmap usa el frame con más detecciones.
                clase: filtra por clase (histograma y heatmap).
                umbral_iou: umbral para metricas_por_clase.
            """
            try:
                return herramientas.graficar(tipo_grafica, frame_id, clase, umbral_iou)
            except ErrorHerramienta as exc:
                return f"Error: {exc}"

        return [buscar_resultados, generar_grafica]


# =============================================================================
# 7. Agente (LangGraph vía langchain.agents.create_agent)
# =============================================================================
SYSTEM_PROMPT = """\
Eres un analista experto en visión por computadora y movilidad urbana. Analizas los
resultados de un detector YOLO11n entrenado sobre imágenes gigapixel (~100-120 MP) del
dataset NJU-HARD: escenas urbanas tomadas desde gran altura, no imágenes satelitales
multiespectrales. Clases: pedestrian (peatón), car (auto), bus (autobús), truck (camión).

Cómo funciona el pipeline, por si te preguntan: cada frame se corta en tiles de 1024 px
con 128 px de solapamiento, YOLO predice en cada tile, las cajas se pasan a coordenadas
del frame y se deduplican con NMS. La posición se describe con una cuadrícula del frame
(celdas rXcY: fila X desde arriba, columna Y desde la izquierda, empezando en 0).

Reglas:
1. Antes de responder sobre datos, usa `buscar_resultados`. No inventes cifras ni frames.
   Para totales usa "resumen_global" o "resumen_frame". Para "¿dónde se concentran...?"
   empieza por "resumen_frame", que ya lista las zonas con más detecciones, y usa
   "region_frame" para el detalle de una celda. La búsqueda es semántica y no ordena por
   cantidad, así que no supongas que el primer resultado es el mayor. Para la calidad
   del modelo filtra por "metricas_modelo".
2. Cita la fuente de cada cifra entre corchetes, tal como aparece en los resultados,
   p. ej. [resumen_frame/seq03_000120] o [metricas_modelo/IoU_0.5/car].
3. Si los documentos recuperados no contienen la respuesta, dilo explícitamente
   ("no tengo esa información en los resultados") y sugiere qué habría que generar.
4. Distingue "detecciones del modelo" de "objetos reales". Con una precisión baja los
   conteos sobreestiman y con un recall bajo subestiman; menciónalo cuando sea relevante.
5. Si un documento advierte que la cobertura es parcial o que las coordenadas son
   aproximadas, trasládalo a la respuesta.
6. Para gráficas usa `generar_grafica` y devuelve la ruta del archivo junto con los datos.
7. Responde en español y sé conciso: primero la respuesta directa y después el detalle.
"""


@dataclass
class Respuesta:
    texto: str
    herramientas_usadas: list[str] = field(default_factory=list)


class AgenteAnalisis:
    """Agente ReAct con memoria de conversación en sesión (por thread_id)."""

    def __init__(self, config: Config, herramientas: Herramientas | None = None, llm=None):
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        self.config = config
        self.herramientas = herramientas or Herramientas(config)
        self.grafo = create_agent(
            llm or crear_llm(config),
            self.herramientas.como_tools(),
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

    def preguntar(self, pregunta: str, thread_id: str = "sesion", al_usar_herramienta=None) -> Respuesta:
        self.herramientas.ultimas_fuentes = []
        usadas: list[str] = []
        texto = ""
        entrada = {"messages": [{"role": "user", "content": pregunta}]}
        configuracion = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

        for paso in self.grafo.stream(entrada, configuracion, stream_mode="updates"):
            for actualizacion in paso.values():
                for mensaje in (actualizacion or {}).get("messages", []):
                    if not isinstance(mensaje, AIMessage):
                        continue
                    for llamada in mensaje.tool_calls:
                        usadas.append(llamada["name"])
                        if al_usar_herramienta:
                            al_usar_herramienta(llamada["name"], llamada["args"])
                    if mensaje.text:
                        texto = mensaje.text
        return Respuesta(texto=texto.strip(), herramientas_usadas=usadas)


# =============================================================================
# 8. Línea de comandos: exportar | indexar | chat
# =============================================================================
AYUDA_CHAT = """\
Comandos:
  /fuentes    documentos recuperados en la última respuesta
  /reindexar  vuelve a indexar resultados/detecciones y metricas.json
  /modelo     proveedor y modelo activos
  /nuevo      empieza una conversación nueva (olvida el contexto)
  /ayuda      muestra esta ayuda
  /salir      termina
"""


def _mostrar_herramienta(nombre: str, args: dict) -> None:
    argumentos = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items() if v is not None)
    print(f"  · {nombre}({argumentos})", flush=True)


def _mostrar_fuentes(fuentes: list[dict]) -> None:
    if not fuentes:
        print("No se recuperaron documentos en la última respuesta.")
        return
    vistos = set()
    for f in fuentes:
        if f["fuente"] not in vistos:
            vistos.add(f["fuente"])
            print(f"- [{f['fuente']}] (distancia {f['distancia']})\n  {f['texto'][:300]}")


def _turno(agente: AgenteAnalisis, pregunta: str, thread_id: str) -> None:
    try:
        respuesta = agente.preguntar(pregunta, thread_id, al_usar_herramienta=_mostrar_herramienta)
    except KeyboardInterrupt:
        print("\n(interrumpido)")
        return
    except Exception as exc:  # red, API key, cuota... no deben cerrar el chat
        mensaje = str(exc)
        print(f"\nError al consultar el LLM: {type(exc).__name__}: {mensaje[:500]}")
        if any(c in mensaje for c in ("503", "504", "529", "UNAVAILABLE", "DEADLINE", "overloaded")):
            print("El servicio del proveedor está saturado. Reintenta en unos minutos, baja"
                  " LLM_EFFORT o cambia LLM_MODEL / LLM_PROVIDER.")
        elif any(c in mensaje for c in ("401", "403", "API key", "api_key", "PERMISSION")):
            print("Revisa la API key del proveedor en .env.")
        elif "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje:
            print("Límite de cuota o de peticiones alcanzado. Espera o revisa tu plan.")
        else:
            print("Revisa la conexión y la configuración del LLM en .env.")
        return
    print(f"\n{respuesta.texto or '(respuesta vacía)'}\n")


def _imprimir_indexado(config: Config, conteo: Counter) -> None:
    eliminados = conteo.pop("_obsoletos_eliminados", 0)
    print(f"Colección '{config.chroma_collection}' en {config.chroma_dir}:")
    for tipo, n in sorted(conteo.items()):
        print(f"  {tipo}: {n}")
    print(f"  total: {sum(conteo.values())} documentos (obsoletos eliminados: {eliminados})")


def chat(config: Config, pregunta: str | None = None) -> None:
    config.validar_llm()
    print(f"Cargando agente ({describir_llm(config)})...", flush=True)
    agente = AgenteAnalisis(config)
    n_docs = agente.herramientas.vectorstore._collection.count()
    if n_docs == 0:
        print("Aviso: la base vectorial está vacía. Ejecuta /reindexar o python agente.py indexar.")

    thread_id = str(uuid.uuid4())
    if pregunta:
        _turno(agente, pregunta, thread_id)
        return

    print(f"{n_docs} documentos indexados. Escribe tu pregunta o /ayuda.\n")
    while True:
        try:
            entrada = input("tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        comando = entrada.lower()
        if not entrada:
            continue
        if comando in {"/salir", "/exit", "/q"}:
            break
        if comando == "/ayuda":
            print(AYUDA_CHAT)
        elif comando == "/fuentes":
            _mostrar_fuentes(agente.herramientas.ultimas_fuentes)
        elif comando == "/modelo":
            print(describir_llm(config))
        elif comando == "/nuevo":
            thread_id = str(uuid.uuid4())
            print("Conversación nueva.")
        elif comando == "/reindexar":
            try:
                _imprimir_indexado(config, indexar(config))
            except Exception as exc:
                print(f"No se pudo reindexar: {exc}")
        elif comando.startswith("/"):
            print("Comando desconocido. /ayuda para ver los disponibles.")
        else:
            _turno(agente, entrada, thread_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agente de análisis YOLO + RAG + LLM.")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_exp = sub.add_parser("exportar", help="Ejecuta YOLO y guarda un JSON por frame.")
    p_exp.add_argument("--fuente", required=True, help="Imagen o carpeta de imágenes.")
    p_exp.add_argument("--pesos", help="Ruta a best.pt (por defecto YOLO_WEIGHTS o runs_nju/).")
    p_exp.add_argument("--modo", choices=["auto", "frames", "tiles"], default="auto")
    p_exp.add_argument("--secuencia", help="ID de secuencia a guardar en los metadatos.")
    p_exp.add_argument("--destino", help="Carpeta de salida (por defecto DETECCIONES_DIR).")

    p_idx = sub.add_parser("indexar", help="Indexa los resultados en ChromaDB.")
    p_idx.add_argument("--reset", action="store_true", help="Reconstruye la colección desde cero.")

    p_chat = sub.add_parser("chat", help="Conversa con el agente.")
    p_chat.add_argument("-p", "--pregunta", help="Hace una sola pregunta y termina.")

    args = parser.parse_args(argv)
    try:
        config = cargar_config()
        if args.comando == "exportar":
            rutas = exportar(
                config, Path(args.fuente).expanduser(), args.pesos, args.modo, args.secuencia,
                Path(args.destino).expanduser() if args.destino else None,
            )
            print(f"{len(rutas)} archivos JSON guardados. Siguiente paso: python agente.py indexar")
        elif args.comando == "indexar":
            print(f"Embeddings: {config.embeddings_model} ({config.embeddings_device})")
            _imprimir_indexado(config, indexar(config, reset=args.reset))
        else:
            chat(config, args.pregunta)
    except ErrorConfiguracion as exc:
        sys.exit(f"Configuración inválida: {exc}")
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
