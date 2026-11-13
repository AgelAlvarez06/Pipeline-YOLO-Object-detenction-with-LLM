from conftest import METRICAS

from agente.documentos import (
    construir_documentos,
    describir_celda,
    documentos_frame,
    documentos_metricas,
)
from agente.exportar_detecciones import celda_de, inferir_split, tile_starts
from pathlib import Path


def test_tile_starts_igual_que_yolo():
    assert tile_starts(1000, 1024, 128) == [0]
    assert tile_starts(3000, 1024, 128) == [0, 896, 1792, 1976]


def test_celda_de_limites():
    assert celda_de(0, 0, 4000, 4000, 4, 4) == "r0c0"
    assert celda_de(3999, 3999, 4000, 4000, 4, 4) == "r3c3"
    assert celda_de(4000, 4000, 4000, 4000, 4, 4) == "r3c3"  # borde exacto no se sale
    assert celda_de(2100, 1100, 4000, 4000, 4, 4) == "r1c2"


def test_inferir_split():
    assert inferir_split(Path("nju_yolo/images/val/a.jpg")) == "val"
    assert inferir_split(Path("x/train/a.jpg")) == "train"
    assert inferir_split(Path("otra/a.jpg")) == "desconocido"


def test_registro_resumen(registro):
    r = registro["resumen"]
    assert r["total"] == 4
    assert r["por_clase"] == {"pedestrian": 3, "car": 1, "bus": 0, "truck": 0}
    assert r["conteo_por_celda"] == {
        "r0c0": {"pedestrian": 2},
        "r1c2": {"car": 1},
        "r3c3": {"pedestrian": 1},
    }
    assert r["confianza_media_por_clase"]["pedestrian"] == round((0.9 + 0.5 + 0.3) / 3, 4)
    # Ordenadas por confianza descendente
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
    config = datos_en_disco
    a = construir_documentos(config.detecciones_dir, config.metricas_path)
    b = construir_documentos(config.detecciones_dir, config.metricas_path)
    assert [d.id for d in a] == [d.id for d in b]
    assert len({d.id for d in a}) == len(a)
    assert a[0].id == "resumen_global"
