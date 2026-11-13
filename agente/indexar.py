"""
Fase 2b: indexa los documentos en ChromaDB con embeddings locales.

Uso:
    python -m agente.indexar          # upsert idempotente (no duplica)
    python -m agente.indexar --reset  # borra la colección y reconstruye
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import lru_cache

# Evita la telemetría de Chroma y el aviso de paralelismo de tokenizers.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agente.config import Config, cargar_config
from agente.documentos import construir_documentos


def _dispositivo(preferido: str) -> str:
    import torch

    if preferido == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if preferido == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return preferido


@lru_cache(maxsize=2)
def crear_embeddings(modelo: str, dispositivo: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=modelo,
        model_kwargs={"device": _dispositivo(dispositivo)},
        encode_kwargs={"normalize_embeddings": True},
    )


def abrir_vectorstore(config: Config) -> Chroma:
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.chroma_collection,
        embedding_function=crear_embeddings(config.embeddings_model, config.embeddings_device),
        persist_directory=str(config.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def indexar(config: Config, reset: bool = False) -> Counter:
    docs = construir_documentos(config.detecciones_dir, config.metricas_path)
    if not docs:
        raise RuntimeError(
            f"No hay nada que indexar: ni JSON en {config.detecciones_dir} ni {config.metricas_path}."
            " Ejecuta antes: python -m agente.exportar_detecciones --fuente <imágenes>"
        )

    store = abrir_vectorstore(config)
    if reset:
        store.delete_collection()
        store = abrir_vectorstore(config)

    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("IDs de documentos duplicados; revisa los frame_id de los JSON.")

    # Elimina documentos de frames que ya no existen (para que reindexar sea exacto).
    existentes = set(store.get(include=[])["ids"])
    obsoletos = sorted(existentes - set(ids))
    if obsoletos:
        store.delete(ids=obsoletos)

    store.add_documents(docs, ids=ids)  # Chroma hace upsert por id
    por_tipo = Counter(d.metadata["tipo"] for d in docs)
    por_tipo["_obsoletos_eliminados"] = len(obsoletos)
    return por_tipo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Indexa los resultados YOLO en ChromaDB.")
    parser.add_argument("--reset", action="store_true", help="Reconstruye la colección desde cero.")
    args = parser.parse_args(argv)

    config = cargar_config()
    print(f"Embeddings: {config.embeddings_model} ({config.embeddings_device})")
    conteo = indexar(config, reset=args.reset)
    eliminados = conteo.pop("_obsoletos_eliminados", 0)
    print(f"Colección '{config.chroma_collection}' en {config.chroma_dir}:")
    for tipo, n in sorted(conteo.items()):
        print(f"  {tipo}: {n}")
    print(f"  total: {sum(conteo.values())} documentos (obsoletos eliminados: {eliminados})")


if __name__ == "__main__":
    main()
