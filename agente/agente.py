"""Fase 4: construcción del agente (LangGraph vía langchain.agents.create_agent)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agente.config import Config
from agente.herramientas import Herramientas
from agente.llm import crear_llm
from agente.prompts import SYSTEM_PROMPT


@dataclass
class Respuesta:
    texto: str
    herramientas_usadas: list[str] = field(default_factory=list)


class AgenteAnalisis:
    """Agente ReAct con memoria de conversación en sesión (por thread_id)."""

    def __init__(self, config: Config, herramientas: Herramientas | None = None, llm=None):
        self.config = config
        self.herramientas = herramientas or Herramientas(config)
        self.grafo = create_agent(
            llm or crear_llm(config),
            self.herramientas.como_tools(),
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

    def preguntar(
        self,
        pregunta: str,
        thread_id: str = "sesion",
        al_usar_herramienta: Callable[[str, dict], None] | None = None,
    ) -> Respuesta:
        self.herramientas.ultimas_fuentes = []
        usadas: list[str] = []
        texto = ""
        entrada = {"messages": [{"role": "user", "content": pregunta}]}
        configuracion = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

        for paso in self.grafo.stream(entrada, configuracion, stream_mode="updates"):
            for actualizacion in paso.values():
                for mensaje in (actualizacion or {}).get("messages", []):
                    if not isinstance(mensaje, AIMessage):
                        continue
                    for llamada in mensaje.tool_calls:
                        usadas.append(llamada["name"])
                        if al_usar_herramienta:
                            al_usar_herramienta(llamada["name"], llamada["args"])
                    if mensaje.text:
                        texto = mensaje.text
        return Respuesta(texto=texto.strip(), herramientas_usadas=usadas)
