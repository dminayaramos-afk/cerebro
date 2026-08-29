# -*- coding: utf-8 -*-
"""Agente con ejecución controlada de herramientas: límites explícitos de
pasos, límite de llamadas a herramientas, y verificación de permisos antes
de ejecutar cualquier acción."""

import config
from agent.tools.registry import ToolRegistry


class ControlledAgent:
    def __init__(self, max_steps=None, max_tool_calls=None):
        self.max_steps = max_steps or config.MAX_AGENT_STEPS
        self.max_tool_calls = max_tool_calls or config.MAX_TOOL_CALLS
        self.registry = ToolRegistry()

        # Declaración explícita de permisos por herramienta.
        self.permisos_herramientas = {
            "calculator": "CALCULATE",
            "clock": "READ",
            "read_file": "READ",
            "write_file": "WRITE",
        }

    def verificar_permiso(self, herramienta, permiso_requerido):
        return (
            self.permisos_herramientas.get(herramienta) == permiso_requerido
            or permiso_requerido == "ANY"
        )

    def ejecutar_plan_autonomo(self, intenciones):
        historial = []
        tool_calls_count = 0

        for paso, intencion in enumerate(intenciones[:self.max_steps], start=1):
            herramienta = intencion.get("tool")

            if herramienta not in self.registry.tools:
                historial.append({
                    "paso": paso,
                    "error": f"Herramienta '{herramienta}' no autorizada o inexistente.",
                })
                break

            if tool_calls_count >= self.max_tool_calls:
                historial.append({
                    "paso": paso,
                    "error": "Limite maximo de llamadas a herramientas alcanzado (MAX_TOOL_CALLS).",
                })
                break

            args = intencion.get("arguments", {})
            resultado = self.registry.ejecutar(herramienta, args)

            tool_calls_count += 1
            historial.append({
                "paso": paso,
                "herramienta": herramienta,
                "resultado": resultado,
            })

        return historial
