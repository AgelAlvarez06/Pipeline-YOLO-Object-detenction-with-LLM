"""
Chat en terminal con el agente de análisis.

Uso:
    python -m agente.chat                       # modo interactivo
    python -m agente.chat -p "¿Cuántos peatones hay?"  # una sola pregunta
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from agente.config import ErrorConfiguracion, cargar_config
from agente.llm import describir_llm

AYUDA = """\
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
        if f["fuente"] in vistos:
            continue
        vistos.add(f["fuente"])
        print(f"- [{f['fuente']}] (distancia {f['distancia']})\n  {f['texto'][:300]}")


def _turno(agente, pregunta: str, thread_id: str) -> None:
    try:
        respuesta = agente.preguntar(pregunta, thread_id, al_usar_herramienta=_mostrar_herramienta)
    except KeyboardInterrupt:
        print("\n(interrumpido)")
        return
    except Exception as exc:  # errores de red, API key, cuota... no deben cerrar el chat
        print(f"\nError al consultar el LLM: {type(exc).__name__}: {exc}")
        print("Revisa la API key / conexión en .env o cambia LLM_PROVIDER.")
        return
    print(f"\n{respuesta.texto or '(respuesta vacía)'}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chat con el agente de análisis YOLO + RAG.")
    parser.add_argument("-p", "--pregunta", help="Hace una sola pregunta y termina.")
    args = parser.parse_args(argv)

    try:
        config = cargar_config()
        config.validar_llm()
    except ErrorConfiguracion as exc:
        sys.exit(f"Configuración inválida: {exc}")

    print(f"Cargando agente ({describir_llm(config)})...", flush=True)
    from agente.agente import AgenteAnalisis

    agente = AgenteAnalisis(config)
    n_docs = agente.herramientas.vectorstore._collection.count()
    if n_docs == 0:
        print("Aviso: la base vectorial está vacía. Ejecuta /reindexar o python -m agente.indexar.")

    thread_id = str(uuid.uuid4())
    if args.pregunta:
        _turno(agente, args.pregunta, thread_id)
        return

    print(f"{n_docs} documentos indexados. Escribe tu pregunta o /ayuda.\n")
    while True:
        try:
            entrada = input("tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not entrada:
            continue
        comando = entrada.lower()
        if comando in {"/salir", "/exit", "/q"}:
            break
        if comando == "/ayuda":
            print(AYUDA)
        elif comando == "/fuentes":
            _mostrar_fuentes(agente.herramientas.ultimas_fuentes)
        elif comando == "/modelo":
            print(describir_llm(config))
        elif comando == "/nuevo":
            thread_id = str(uuid.uuid4())
            print("Conversación nueva.")
        elif comando == "/reindexar":
            from agente.indexar import indexar

            try:
                conteo = indexar(config)
                conteo.pop("_obsoletos_eliminados", None)
                print(f"Reindexado: {dict(conteo)}")
            except Exception as exc:
                print(f"No se pudo reindexar: {exc}")
        elif comando.startswith("/"):
            print("Comando desconocido. /ayuda para ver los disponibles.")
        else:
            _turno(agente, entrada, thread_id)


if __name__ == "__main__":
    main()
