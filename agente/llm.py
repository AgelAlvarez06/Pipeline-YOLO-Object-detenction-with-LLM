"""Fábrica del modelo de chat según LLM_PROVIDER (anthropic | openai | google_genai | ollama)."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agente.config import Config


def crear_llm(config: Config) -> BaseChatModel:
    config.validar_llm()
    proveedor = config.llm_provider
    kwargs: dict = {}

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
        # Gemini (Google AI Studio). La clave se lee de GOOGLE_API_KEY.
        # No se envía temperature: Google recomienda dejar el valor por defecto
        # (1.0) en Gemini 3, porque bajarlo puede provocar bucles o respuestas peores.
        kwargs["max_output_tokens"] = config.llm_max_tokens
    elif proveedor == "ollama":
        kwargs["base_url"] = config.ollama_base_url
        kwargs["num_predict"] = config.llm_max_tokens
        if config.llm_temperature is not None:
            kwargs["temperature"] = config.llm_temperature

    return init_chat_model(config.llm_model, model_provider=proveedor, **kwargs)


def describir_llm(config: Config) -> str:
    extra = f", effort={config.llm_effort}" if config.llm_provider == "anthropic" and config.llm_effort else ""
    return f"{config.llm_provider} / {config.llm_model}{extra}"
