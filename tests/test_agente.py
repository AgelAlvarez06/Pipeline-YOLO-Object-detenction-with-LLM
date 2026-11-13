"""Tests de agente.py. No usan LLM real ni red:  pytest tests"""

import json
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agente  # noqa: E402
from agente import (  # noqa: E402
    AgenteAnalisis,
    ErrorConfiguracion,
    Herramientas,
    cargar_config,
    celda_de,
    construir_documentos,
    construir_registro,
    crear_llm,
    describir_celda,
    documentos_frame,
    documentos_metricas,
    inferir_split,
    tile_starts,
)

# =============================================================================
# Datos de prueba
# =============================================================================
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
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "YOLO_WEIGHTS",
    "GRID_FILAS", "GRID_COLUMNAS",
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
    return cargar_config(env_path=tmp_path / "sin.env")  # no carga el .env real


def _det(clase, clase_id, conf, x, y):
    return {"clase": clase, "clase_id": clase_id, "confianza": conf, "bbox_xyxy": [x, y, x + 20, y + 40]}


@pytest.fixture
def registro(config):
    # Frame de 4000x4000 con cuadrícula 4x4 (celdas de 1000 px)
    detecciones = [
        _det("pedestrian", 0, 0.9, 100, 100),    # r0c0
        _det("pedestrian", 0, 0.5, 150, 120),    # r0c0
        _det("pedestrian", 0, 0.3, 3500, 3500),  # r3c3
        _det("car", 1, 0.4, 2100, 1100),         # r1c2
    ]
    return construir_registro(
        frame_id="seq01_000001", detecciones=detecciones, ancho=4000, alto=4000,
        ruta_imagen="frame.jpg", split="val", secuencia="seq01",
        modelo={"pesos": "best.pt", "conf": 0.25, "imgsz": 640, "tile_size": 1024, "overlap": 128, "dedup_iou": 0.6},
        config=config,
    )


@pytest.fixture
def datos_en_disco(config, registro):
    config.detecciones_dir.mkdir(parents=True)
    (config.detecciones_dir / "seq01_000001.json").write_text(json.dumps(registro), encoding="utf-8")
    config.metricas_path.write_text(json.dumps(METRICAS), encoding="utf-8")
    return config


@pytest.fixture
def embeddings_falsos(monkeypatch):
    falsos = DeterministicFakeEmbedding(size=64)
    monkeypatch.setattr(agente, "crear_embeddings", lambda *_: falsos)


# =============================================================================
# Exportación y documentos
# =============================================================================
def test_tile_starts_igual_que_yolo():
    assert tile_starts(1000, 1024, 128) == [0]
    assert tile_starts(3000, 1024, 128) == [0, 896, 1792, 1976]


def test_celda_de_limites():
    assert celda_de(0, 0, 4000, 4000, 4, 4) == "r0c0"
    assert celda_de(3999, 3999, 4000, 4000, 4, 4) == "r3c3"
    assert celda_de(4000, 4000, 4000, 4000, 4, 4) == "r3c3"  # el borde exacto no se sale
    assert celda_de(2100, 1100, 4000, 4000, 4, 4) == "r1c2"


def test_inferir_split():
    assert inferir_split(Path("nju_yolo/images/val/a.jpg")) == "val"
    assert inferir_split(Path("x/train/a.jpg")) == "train"
    assert inferir_split(Path("otra/a.jpg")) == "desconocido"


def test_registro_resumen(registro):
    r = registro["resumen"]
    assert r["total"] == 4
    assert r["por_clase"] == {"pedestrian": 3, "car": 1, "bus": 0, "truck": 0}
    assert r["conteo_por_celda"] == {"r0c0": {"pedestrian": 2}, "r1c2": {"car": 1}, "r3c3": {"pedestrian": 1}}
    assert r["confianza_media_por_clase"]["pedestrian"] == round((0.9 + 0.5 + 0.3) / 3, 4)
    assert [d["confianza"] for d in registro["detecciones"]] == [0.9, 0.5, 0.4, 0.3]


def test_describir_celda():
    assert describir_celda("r0c0", 4, 4).endswith("(zona superior-izquierda)")
    assert describir_celda("r3c3", 4, 4).endswith("(zona inferior-derecha)")
    assert "zona central" in describir_celda("r1c1", 3, 3)


def test_documentos_frame(registro):
    docs = documentos_frame(registro)
    ids = [d.id for d in docs]
    assert ids[0] == "resumen_frame/seq01_000001"
    assert set(ids[1:]) == {f"region_frame/seq01_000001/{c}" for c in ("r0c0", "r1c2", "r3c3")}
    resumen = docs[0].page_content
    assert "4 objetos" in resumen and "3 peatones" in resumen and "1 autos" in resumen
    for d in docs:
        assert d.metadata["fuente"] == d.id
        assert all(v is not None for v in d.metadata.values())


def test_metricas_interpreta_falsos_positivos():
    docs = {d.id: d for d in documentos_metricas(METRICAS)}
    assert "metricas_modelo/IoU_0.5" in docs
    car = docs["metricas_modelo/IoU_0.5/car"].page_content
    assert "falsos positivos" in car and "indefinido" in car
    assert "no aplican" in docs["metricas_modelo/IoU_0.5/bus"].page_content


def test_ids_deterministas_y_unicos(datos_en_disco):
    a = construir_documentos(datos_en_disco.detecciones_dir, datos_en_disco.metricas_path)
    b = construir_documentos(datos_en_disco.detecciones_dir, datos_en_disco.metricas_path)
    assert [d.id for d in a] == [d.id for d in b]
    assert len({d.id for d in a}) == len(a)
    assert a[0].id == "resumen_global"


# =============================================================================
# Herramientas, índice y gráficas
# =============================================================================
@pytest.mark.parametrize(
    "tipo,kwargs",
    [
        ("barras_clases", {}),
        ("barras_clases", {"frame_id": "seq01_000001"}),
        ("histograma_confianza", {"clase": "pedestrian"}),
        ("heatmap_celdas", {}),
        ("metricas_por_clase", {}),
    ],
)
def test_generar_grafica_crea_png(datos_en_disco, tipo, kwargs):
    salida = Herramientas(datos_en_disco).graficar(tipo, **kwargs)
    ruta = Path(salida.split("Gráfica guardada en: ")[1].splitlines()[0])
    assert ruta.suffix == ".png" and ruta.stat().st_size > 1000


def test_heatmap_datos_exactos(datos_en_disco):
    salida = Herramientas(datos_en_disco).graficar("heatmap_celdas", clase="pedestrian")
    assert "[[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]" in salida


def test_tool_devuelve_error_legible(datos_en_disco):
    tools = {t.name: t for t in Herramientas(datos_en_disco).como_tools()}
    salida = tools["generar_grafica"].invoke({"tipo_grafica": "heatmap_celdas", "frame_id": "no_existe"})
    assert salida.startswith("Error:") and "seq01_000001" in salida


def test_grafica_sin_detecciones(config):
    tools = {t.name: t for t in Herramientas(config).como_tools()}
    assert "No hay detecciones" in tools["generar_grafica"].invoke({"tipo_grafica": "barras_clases"})


def test_indexar_es_idempotente(datos_en_disco, embeddings_falsos):
    primero = agente.indexar(datos_en_disco)
    segundo = agente.indexar(datos_en_disco)
    total = sum(v for k, v in primero.items() if not k.startswith("_"))
    assert agente.abrir_vectorstore(datos_en_disco)._collection.count() == total
    assert segundo["_obsoletos_eliminados"] == 0


def test_buscar_con_filtros(datos_en_disco, embeddings_falsos):
    agente.indexar(datos_en_disco)
    h = Herramientas(datos_en_disco)
    salida = h.buscar("peatones por zona", tipo="region_frame", frame_id="seq01_000001")
    assert "[fuente: region_frame/seq01_000001/" in salida
    assert "metricas_modelo" not in salida
    assert h.ultimas_fuentes
    with pytest.raises(ValueError):
        h.buscar("x", tipo="inventado")


# =============================================================================
# Configuración y LLM
# =============================================================================
def test_config_sin_api_key(config):
    with pytest.raises(ErrorConfiguracion, match="ANTHROPIC_API_KEY"):
        config.validar_llm()


def test_config_ollama_no_necesita_clave(config, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama")
    cargar_config(env_path=tmp_path / "sin.env").validar_llm()


def test_config_proveedor_invalido(config, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "otro")
    with pytest.raises(ErrorConfiguracion, match="no soportado"):
        cargar_config(env_path=tmp_path / "sin.env").validar_llm()


def test_llm_anthropic_sin_temperature(config, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-prueba")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("LLM_EFFORT", "medium")
    llm = crear_llm(cargar_config(env_path=tmp_path / "sin.env"))
    assert llm.temperature is None  # los modelos Claude actuales rechazan temperature
    assert llm.output_config == {"effort": "medium"}


def test_llm_gemini(config, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")  # alias de google_genai
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4000")
    monkeypatch.setenv("LLM_EFFORT", "max")
    cfg = cargar_config(env_path=tmp_path / "sin.env")
    assert cfg.llm_provider == "google_genai"
    with pytest.raises(ErrorConfiguracion, match="GOOGLE_API_KEY"):
        cfg.validar_llm()

    monkeypatch.setenv("GOOGLE_API_KEY", "clave-prueba")
    llm = crear_llm(cfg)
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.model.endswith("gemini-3.5-flash")
    assert llm.max_output_tokens == 4000
    assert llm.thinking_level == "high"  # "max" no existe en Gemini: se mapea a "high"
    assert llm.timeout == 120 and llm.max_retries == 2  # para no quedarse colgado


# =============================================================================
# Agente (LLM falso)
# =============================================================================
class ModeloFalso(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class VectorstoreFalso:
    def similarity_search_with_score(self, consulta, k, filter=None):
        doc = Document(page_content="El modelo detectó 3 peatones.", metadata={"fuente": "resumen_global"})
        return [(doc, 0.1)]


def test_agente_usa_herramienta_y_responde(config):
    mensajes = iter([
        AIMessage(content="", tool_calls=[{"name": "buscar_resultados", "args": {"consulta": "peatones"}, "id": "t1"}]),
        AIMessage(content="Se detectaron 3 peatones [resumen_global]."),
    ])
    ag = AgenteAnalisis(config, Herramientas(config, VectorstoreFalso()), llm=ModeloFalso(messages=mensajes))
    vistas = []
    r = ag.preguntar("¿Cuántos peatones?", al_usar_herramienta=lambda n, a: vistas.append(n))
    assert r.texto == "Se detectaron 3 peatones [resumen_global]."
    assert r.herramientas_usadas == ["buscar_resultados"] == vistas
    assert ag.herramientas.ultimas_fuentes[0]["fuente"] == "resumen_global"
