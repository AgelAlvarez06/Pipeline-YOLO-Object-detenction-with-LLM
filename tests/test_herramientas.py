import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agente import indexar as modulo_indexar
from agente.config import ErrorConfiguracion
from agente.herramientas import Herramientas


# ------------------------------------------------------------------- gráficas
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
    h = Herramientas(datos_en_disco)
    salida = h.graficar(tipo, **kwargs)
    ruta = salida.split("Gráfica guardada en: ")[1].splitlines()[0]
    assert ruta.endswith(".png")
    assert (datos_en_disco.graficas_dir / ruta.split("/")[-1]).stat().st_size > 1000


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


# ------------------------------------------------------------- RAG / índice
@pytest.fixture
def embeddings_falsos(monkeypatch):
    falsos = DeterministicFakeEmbedding(size=64)
    monkeypatch.setattr(modulo_indexar, "crear_embeddings", lambda *_: falsos)


def test_indexar_es_idempotente(datos_en_disco, embeddings_falsos):
    primero = modulo_indexar.indexar(datos_en_disco)
    segundo = modulo_indexar.indexar(datos_en_disco)
    total = sum(v for k, v in primero.items() if not k.startswith("_"))
    store = modulo_indexar.abrir_vectorstore(datos_en_disco)
    assert store._collection.count() == total
    assert segundo["_obsoletos_eliminados"] == 0


def test_buscar_con_filtros(datos_en_disco, embeddings_falsos):
    modulo_indexar.indexar(datos_en_disco)
    h = Herramientas(datos_en_disco)
    salida = h.buscar("peatones por zona", tipo="region_frame", frame_id="seq01_000001")
    assert "[fuente: region_frame/seq01_000001/" in salida
    assert "metricas_modelo" not in salida
    assert h.ultimas_fuentes
    with pytest.raises(ValueError):
        h.buscar("x", tipo="inventado")


# ------------------------------------------------------------ config / LLM
def test_config_sin_api_key(config):
    with pytest.raises(ErrorConfiguracion, match="ANTHROPIC_API_KEY"):
        config.validar_llm()


def test_config_ollama_no_necesita_clave(config, monkeypatch, tmp_path):
    from agente.config import cargar_config

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama")
    cargar_config(env_path=tmp_path / "sin.env").validar_llm()


def test_config_proveedor_invalido(monkeypatch, tmp_path):
    from agente.config import cargar_config

    monkeypatch.setenv("LLM_PROVIDER", "otro")
    with pytest.raises(ErrorConfiguracion, match="no soportado"):
        cargar_config(env_path=tmp_path / "sin.env").validar_llm()


def test_llm_anthropic_sin_temperature(monkeypatch, tmp_path):
    from agente.config import cargar_config
    from agente.llm import crear_llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-prueba")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("LLM_EFFORT", "medium")
    llm = crear_llm(cargar_config(env_path=tmp_path / "sin.env"))
    assert llm.temperature is None  # los modelos Claude actuales rechazan temperature
    assert llm.output_config == {"effort": "medium"}


def test_llm_gemini(monkeypatch, tmp_path):
    from agente.config import cargar_config
    from agente.llm import crear_llm

    monkeypatch.setenv("LLM_PROVIDER", "gemini")  # alias de google_genai
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4000")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = cargar_config(env_path=tmp_path / "sin.env")
    assert config.llm_provider == "google_genai"
    with pytest.raises(ErrorConfiguracion, match="GOOGLE_API_KEY"):
        config.validar_llm()

    monkeypatch.setenv("GOOGLE_API_KEY", "clave-prueba")
    llm = crear_llm(config)
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.model.endswith("gemini-3.5-flash")
    assert llm.max_output_tokens == 4000


# ------------------------------------------------------------------- agente
class ModeloFalso(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class VectorstoreFalso:
    def similarity_search_with_score(self, consulta, k, filter=None):
        doc = Document(page_content="El modelo detectó 3 peatones.", metadata={"fuente": "resumen_global"})
        return [(doc, 0.1)]


def test_agente_usa_herramienta_y_responde(config):
    from agente.agente import AgenteAnalisis

    mensajes = iter(
        [
            AIMessage(content="", tool_calls=[{"name": "buscar_resultados", "args": {"consulta": "peatones"}, "id": "t1"}]),
            AIMessage(content="Se detectaron 3 peatones [resumen_global]."),
        ]
    )
    agente = AgenteAnalisis(config, Herramientas(config, VectorstoreFalso()), llm=ModeloFalso(messages=mensajes))
    vistas = []
    r = agente.preguntar("¿Cuántos peatones?", al_usar_herramienta=lambda n, a: vistas.append(n))
    assert r.texto == "Se detectaron 3 peatones [resumen_global]."
    assert r.herramientas_usadas == ["buscar_resultados"] == vistas
    assert agente.herramientas.ultimas_fuentes[0]["fuente"] == "resumen_global"
