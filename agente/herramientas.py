"""
Fase 3: herramientas que el agente puede llamar.

- buscar_resultados: búsqueda semántica en ChromaDB (RAG).
- generar_grafica: lee los JSON / metricas.json directamente (no el RAG) y
  genera un PNG con matplotlib, así los números de la gráfica son exactos.
"""

from __future__ import annotations

import json
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from langchain.tools import tool  # noqa: E402

from agente.config import Config  # noqa: E402
from agente.documentos import TIPOS, cargar_registros, nombre_es  # noqa: E402

TipoGrafica = Literal[
    "barras_clases", "histograma_confianza", "heatmap_celdas", "metricas_por_clase"
]


class ErrorHerramienta(ValueError):
    """Error con mensaje legible que se devuelve al LLM."""


class Herramientas:
    """Agrupa las herramientas y el estado compartido (vectorstore, últimas fuentes)."""

    def __init__(self, config: Config, vectorstore=None):
        self.config = config
        self._vectorstore = vectorstore
        self.ultimas_fuentes: list[dict] = []

    # ------------------------------------------------------------------ RAG
    @property
    def vectorstore(self):
        if self._vectorstore is None:
            from agente.indexar import abrir_vectorstore

            self._vectorstore = abrir_vectorstore(self.config)
        return self._vectorstore

    def buscar(self, consulta: str, tipo: str | None = None, frame_id: str | None = None) -> str:
        if tipo and tipo not in TIPOS:
            raise ErrorHerramienta(f"tipo '{tipo}' no válido. Opciones: {', '.join(TIPOS)}.")
        condiciones = [{k: v} for k, v in (("tipo", tipo), ("frame_id", frame_id)) if v]
        filtro = None
        if len(condiciones) == 1:
            filtro = condiciones[0]
        elif condiciones:
            filtro = {"$and": condiciones}

        resultados = self.vectorstore.similarity_search_with_score(
            consulta, k=self.config.rag_top_k, filter=filtro
        )
        self.ultimas_fuentes.extend(
            {"fuente": d.metadata.get("fuente"), "distancia": round(s, 4), "texto": d.page_content}
            for d, s in resultados
        )
        if not resultados:
            ayuda = " Prueba sin filtros." if filtro else " ¿Está indexado? (python -m agente.indexar)"
            return "No se encontraron documentos para esa consulta." + ayuda
        return "\n\n".join(f"[fuente: {d.metadata.get('fuente')}]\n{d.page_content}" for d, _ in resultados)

    # ------------------------------------------------------------- Gráficas
    def _registros(self) -> list[dict]:
        registros = cargar_registros(self.config.detecciones_dir)
        if not registros:
            raise ErrorHerramienta(
                f"No hay detecciones en {self.config.detecciones_dir}. "
                "Hay que ejecutar antes agente.exportar_detecciones."
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

    def graficar(
        self,
        tipo_grafica: str,
        frame_id: str | None = None,
        clase: str | None = None,
        umbral_iou: str = "IoU_0.5",
    ) -> str:
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
            r = self._registro(frame_id)
            conteo, titulo = r["resumen"]["por_clase"], f"Detecciones por clase, frame {frame_id}"
        else:
            conteo = {}
            for r in self._registros():
                for c, n in r["resumen"]["por_clase"].items():
                    conteo[c] = conteo.get(c, 0) + n
            titulo = "Detecciones por clase (todos los frames)"
        etiquetas = [nombre_es(c) for c in conteo]
        barras = ax.bar(etiquetas, list(conteo.values()), color="#2a78d6")
        ax.bar_label(barras)
        ax.set(title=titulo, xlabel="Clase", ylabel="Número de detecciones")
        return f"barras_clases_{frame_id or 'global'}", json.dumps(conteo, ensure_ascii=False)

    def _histograma_confianza(self, ax, frame_id, clase, umbral):
        registros = [self._registro(frame_id)] if frame_id else self._registros()
        valores = [
            d["confianza"]
            for r in registros
            for d in r["detecciones"]
            if clase is None or d["clase"] == clase
        ]
        if not valores:
            raise ErrorHerramienta(f"No hay detecciones de '{clase}' para graficar.")
        ax.hist(valores, bins=20, range=(0, 1), color="#2a78d6", edgecolor="white")
        ax.axvline(self.config.yolo_conf, color="#d03b3b", linestyle="--", label="umbral conf")
        ax.legend()
        sujeto = nombre_es(clase) if clase else "todas las clases"
        ax.set(
            title=f"Confianza de las detecciones ({sujeto}, {frame_id or 'todos los frames'})",
            xlabel="Confianza",
            ylabel="Número de detecciones",
        )
        media = sum(valores) / len(valores)
        resumen = f"n={len(valores)}, media={media:.3f}, min={min(valores):.3f}, max={max(valores):.3f}"
        return f"histograma_confianza_{frame_id or 'global'}_{clase or 'todas'}", resumen

    def _heatmap_celdas(self, ax, frame_id, clase, umbral):
        r = self._registro(frame_id)
        filas, cols = r["grid"]["filas"], r["grid"]["columnas"]
        matriz = [[0] * cols for _ in range(filas)]
        for celda, conteo in r["resumen"]["conteo_por_celda"].items():
            f, c = int(celda[1 : celda.index("c")]), int(celda[celda.index("c") + 1 :])
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
        metricas = json.loads(self.config.metricas_path.read_text(encoding="utf-8"))
        por_umbral = metricas.get("por_umbral", {})
        if umbral not in por_umbral:
            raise ErrorHerramienta(f"umbral_iou '{umbral}' no existe. Opciones: {', '.join(por_umbral)}.")
        por_clase = por_umbral[umbral]["por_clase"]
        clases = list(por_clase)
        ancho = 0.27
        for i, (metrica, color) in enumerate(
            (("precision", "#2a78d6"), ("recall", "#e08a1e"), ("F1", "#3a9b5c"))
        ):
            xs = [j + (i - 1) * ancho for j in range(len(clases))]
            ax.bar(xs, [por_clase[c][metrica] for c in clases], ancho, label=metrica, color=color)
        ax.set_xticks(range(len(clases)), [nombre_es(c) for c in clases])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set(title=f"Métricas por clase ({umbral})", xlabel="Clase", ylabel="Valor")
        resumen = {
            c: {k: por_clase[c][k] for k in ("TP", "FP", "FN", "precision", "recall", "F1")}
            for c in clases
        }
        return f"metricas_por_clase_{umbral}", json.dumps(resumen, ensure_ascii=False)

    # --------------------------------------------------- Tools para LangChain
    def como_tools(self) -> list:
        herramientas = self

        @tool
        def buscar_resultados(
            consulta: str, tipo: str | None = None, frame_id: str | None = None
        ) -> str:
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
