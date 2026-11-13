"""Carga y valida la configuración del agente desde el archivo .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent

# Mismas clases que yolo.py. Se duplican aquí para no importar yolo.py,
# que al importarse carga torch/datasets y exige GPU NVIDIA.
CLASS_NAMES = {0: "pedestrian", 1: "car", 2: "bus", 3: "truck"}
NOMBRES_ES = {
    "pedestrian": "peatones",
    "car": "autos",
    "bus": "autobuses",
    "truck": "camiones",
}

# Proveedor -> variable de entorno con la API key (None = no necesita clave)
PROVEEDORES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",  # Gemini con API key de Google AI Studio
    "ollama": None,
}
# Nombres alternativos aceptados en LLM_PROVIDER
ALIAS_PROVEEDOR = {"gemini": "google_genai", "google": "google_genai"}
NIVELES_EFFORT = {"low", "medium", "high", "xhigh", "max"}


class ErrorConfiguracion(RuntimeError):
    """Configuración inválida o incompleta en .env."""


def _leer(nombre: str, defecto: str = "") -> str:
    valor = os.getenv(nombre)
    if valor is None or not valor.strip():
        return defecto
    return valor.strip()


def _numero(nombre: str, defecto: str, tipo):
    valor = _leer(nombre, defecto)
    try:
        return tipo(valor)
    except ValueError as exc:
        raise ErrorConfiguracion(
            f"{nombre}={valor!r} no es un {tipo.__name__} válido."
        ) from exc


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
                f"Opciones: {', '.join(PROVEEDORES)}."
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
        ollama_base_url=_leer("OLLAMA_BASE_URL", "http://localhost:11434"),
        embeddings_model=_leer(
            "EMBEDDINGS_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
