"""
Fase 2a: convierte los JSON de detecciones y metricas.json en documentos para el RAG.

Las cifras se precalculan aquí y se escriben en el texto, para que el LLM las cite
en lugar de contar cajas por su cuenta. Cada documento lleva un `fuente` estable
(p. ej. "resumen_frame/seq03_000120") que el agente usa como cita.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from agente.config import NOMBRES_ES

TIPOS = ("resumen_global", "resumen_frame", "region_frame", "metricas_modelo")


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------
def nombre_es(clase: str) -> str:
    return NOMBRES_ES.get(clase, clase)


def _tercio(indice: int, total: int, nombres: tuple[str, str, str]) -> str:
    posicion = (indice + 0.5) / total
    if posicion < 1 / 3:
        return nombres[0]
    if posicion < 2 / 3:
        return nombres[1]
    return nombres[2]


def describir_celda(celda: str, filas: int, columnas: int) -> str:
    """'r0c3' -> 'fila 1 de 4, columna 4 de 4 (zona superior-derecha)'."""
    fila = int(celda[1 : celda.index("c")])
    col = int(celda[celda.index("c") + 1 :])
    vertical = _tercio(fila, filas, ("superior", "centro", "inferior"))
    horizontal = _tercio(col, columnas, ("izquierda", "centro", "derecha"))
    zona = "central" if vertical == horizontal == "centro" else f"{vertical}-{horizontal}"
    return f"fila {fila + 1} de {filas}, columna {col + 1} de {columnas} (zona {zona})"


def _lista_conteos(conteo: dict[str, int]) -> str:
    partes = [f"{n} {nombre_es(c)}" for c, n in conteo.items() if n]
    return ", ".join(partes) if partes else "ninguna detección"


def _metadatos(**valores) -> dict:
    """Chroma solo admite str/int/float/bool: se descartan los None."""
    return {k: v for k, v in valores.items() if v is not None}


def _doc(texto: str, id_: str, **meta) -> Document:
    return Document(page_content=texto, id=id_, metadata=_metadatos(fuente=id_, **meta))


# ---------------------------------------------------------------------------
# Detecciones por frame
# ---------------------------------------------------------------------------
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
    base = _metadatos(
        frame_id=frame_id, secuencia=registro.get("secuencia"), split=registro.get("split")
    )

    conf_media = resumen.get("confianza_media_por_clase", {})
    confs = ", ".join(f"{nombre_es(c)} {v:.2f}" for c, v in conf_media.items()) or "n/d"

    celdas = resumen.get("conteo_por_celda", {})
    top = sorted(celdas.items(), key=lambda kv: -sum(kv[1].values()))[:3]
    top_txt = "; ".join(
        f"{describir_celda(c, filas, cols)}: {sum(v.values())} ({_lista_conteos(v)})"
        for c, v in top
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
        f" deduplicación IoU {modelo.get('dedup_iou')}, pesos {modelo.get('pesos')}."
        + _nota_cobertura(registro)
    )
    docs = [
        _doc(
            texto,
            f"resumen_frame/{frame_id}",
            tipo="resumen_frame",
            total=resumen["total"],
            **base,
        )
    ]

    confs_celda: dict[str, dict[str, list[float]]] = {}
    for det in registro.get("detecciones", []):
        confs_celda.setdefault(det["celda"], {}).setdefault(det["clase"], []).append(
            det["confianza"]
        )

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
            " del frame." + _nota_cobertura(registro)
        )
        docs.append(
            _doc(
                texto,
                f"region_frame/{frame_id}/{celda}",
                tipo="region_frame",
                celda=celda,
                total=total,
                **base,
            )
        )
    return docs


def documento_global(registros: list[dict]) -> Document:
    totales: Counter = Counter()
    for r in registros:
        totales.update(r["resumen"]["por_clase"])
    ranking = sorted(registros, key=lambda r: -r["resumen"]["total"])
    top = "; ".join(f"{r['frame_id']} ({r['resumen']['total']})" for r in ranking[:5])
    vehiculos = sorted(
        registros,
        key=lambda r: -sum(r["resumen"]["por_clase"].get(c, 0) for c in ("car", "bus", "truck")),
    )[0]
    n_vehiculos = sum(vehiculos["resumen"]["por_clase"].get(c, 0) for c in ("car", "bus", "truck"))
    total = sum(totales.values())
    reparto = ", ".join(
        f"{nombre_es(c)} {n} ({100 * n / max(1, total):.1f}%)" for c, n in totales.items()
    )
    texto = (
        f"Resumen global de {len(registros)} frames analizados: {total} detecciones en total."
        f" Distribución por clase: {reparto}."
        f" Frames con más detecciones: {top}."
        f" Frame con más vehículos (autos+autobuses+camiones): {vehiculos['frame_id']}"
        f" con {n_vehiculos}."
        f" Frames indexados: {', '.join(r['frame_id'] for r in registros)}."
    )
    return _doc(texto, "resumen_global", tipo="resumen_global", n_frames=len(registros))


# ---------------------------------------------------------------------------
# Métricas del modelo (metricas.json)
# ---------------------------------------------------------------------------
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
    docs = []
    dedup = metricas.get("cajas_predichas_deduplicacion", {})
    dedup_txt = ", ".join(
        f"{nombre_es(c)} {v['antes']}→{v['despues']}" for c, v in dedup.items() if v.get("antes")
    )
    contexto = (
        f"Evaluación sobre {metricas.get('frames_evaluados', '?')} frame(s) completo(s),"
        f" deduplicación entre tiles con IoU {metricas.get('dedup_iou', '?')}"
        + (f" (cajas antes→después: {dedup_txt})" if dedup_txt else "")
        + "."
    )

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
            docs.append(
                _doc(
                    texto,
                    f"metricas_modelo/{umbral}/{clase}",
                    tipo="metricas_modelo",
                    umbral_iou=umbral,
                    clase=clase,
                )
            )
    return docs


# ---------------------------------------------------------------------------
# Carga desde disco
# ---------------------------------------------------------------------------
def cargar_registros(detecciones_dir: Path) -> list[dict]:
    if not detecciones_dir.exists():
        return []
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(detecciones_dir.glob("*.json"))
    ]


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
