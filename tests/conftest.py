import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agente.config import cargar_config  # noqa: E402
from agente.exportar_detecciones import construir_registro  # noqa: E402

METRICAS = {
    "frames_evaluados": 1,
    "dedup_iou": 0.6,
    "cajas_predichas_deduplicacion": {
        "pedestrian": {"antes": 680, "despues": 40},
        "car": {"antes": 342, "despues": 9},
    },
    "por_umbral": {
        "IoU_0.5": {
            "por_clase": {
                "pedestrian": {"TP": 8, "FP": 32, "FN": 18, "precision": 0.2, "recall": 0.3077, "F1": 0.2424, "IoU_medio_TP": 0.58},
                "car": {"TP": 0, "FP": 9, "FN": 0, "precision": 0.0, "recall": 0.0, "F1": 0.0, "IoU_medio_TP": None},
                "bus": {"TP": 0, "FP": 0, "FN": 0, "precision": 0.0, "recall": 0.0, "F1": 0.0, "IoU_medio_TP": None},
            },
            "micro": {"TP": 8, "FP": 41, "FN": 18, "precision": 0.1633, "recall": 0.3077, "F1": 0.2133, "IoU_medio_TP": 0.58},
            "macro": {"precision": 0.2, "recall": 0.3077, "F1": 0.2424},
        }
    },
}

VARIABLES = [
    "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_EFFORT",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "YOLO_WEIGHTS", "GRID_FILAS", "GRID_COLUMNAS",
]


@pytest.fixture
def config(tmp_path, monkeypatch):
    for var in VARIABLES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DETECCIONES_DIR", str(tmp_path / "detecciones"))
    monkeypatch.setenv("METRICAS_PATH", str(tmp_path / "metricas.json"))
    monkeypatch.setenv("GRAFICAS_DIR", str(tmp_path / "graficas"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "test")
    # Ruta inexistente: no se carga el .env real del proyecto
    return cargar_config(env_path=tmp_path / "sin.env")


def _det(clase, clase_id, conf, x, y):
    return {"clase": clase, "clase_id": clase_id, "confianza": conf, "bbox_xyxy": [x, y, x + 20, y + 40]}


@pytest.fixture
def registro(config):
    # Frame de 4000x4000 con cuadrícula 4x4 (celdas de 1000 px)
    detecciones = [
        _det("pedestrian", 0, 0.9, 100, 100),     # r0c0
        _det("pedestrian", 0, 0.5, 150, 120),     # r0c0
        _det("pedestrian", 0, 0.3, 3500, 3500),   # r3c3
        _det("car", 1, 0.4, 2100, 1100),          # r1c2
    ]
    return construir_registro(
        frame_id="seq01_000001",
        detecciones=detecciones,
        ancho=4000,
        alto=4000,
        ruta_imagen="frame.jpg",
        split="val",
        secuencia="seq01",
        modelo={"pesos": "best.pt", "conf": 0.25, "imgsz": 640, "tile_size": 1024, "overlap": 128, "dedup_iou": 0.6},
        config=config,
    )


@pytest.fixture
def datos_en_disco(config, registro):
    config.detecciones_dir.mkdir(parents=True)
    (config.detecciones_dir / "seq01_000001.json").write_text(json.dumps(registro), encoding="utf-8")
    config.metricas_path.write_text(json.dumps(METRICAS), encoding="utf-8")
    return config
