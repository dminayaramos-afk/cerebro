# -*- coding: utf-8 -*-
"""Registro de herramientas del agente. La calculadora usa un evaluador
aritmético propio basado en `ast` (no `eval`), para no ejecutar código
Python arbitrario aunque el filtro de caracteres previo fallara."""

import ast
import datetime
import operator
import os

from config import WORKSPACE_DIR

_OPERADORES_PERMITIDOS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluar_nodo(nodo):
    if isinstance(nodo, ast.Constant):
        if isinstance(nodo.value, (int, float)):
            return nodo.value
        raise ValueError("Solo se permiten numeros.")
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES_PERMITIDOS:
        return _OPERADORES_PERMITIDOS[type(nodo.op)](
            _evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right)
        )
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES_PERMITIDOS:
        return _OPERADORES_PERMITIDOS[type(nodo.op)](_evaluar_nodo(nodo.operand))
    raise ValueError("Expresion no permitida.")


def evaluar_expresion_segura(expresion: str):
    """Evalúa una expresión aritmética simple (+ - * / ** paréntesis) sin
    usar eval(): se parsea a un AST y solo se permiten nodos numéricos y
    operadores aritméticos explícitamente listados."""
    arbol = ast.parse(expresion, mode="eval")
    return _evaluar_nodo(arbol.body)


class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.registrar_base()

    def registrar(self, nombre, desc, func, schema):
        self.tools[nombre] = {"desc": desc, "func": func, "schema": schema}

    def registrar_base(self):
        def calc(expression):
            try:
                return {"result": evaluar_expresion_segura(expression)}
            except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
                return {"error": f"Expresion invalida: {e}"}
        self.registrar("calculator", "Calculadora matematica segura", calc, {"expression": str})

        def clock():
            return {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        self.registrar("clock", "Hora local del sistema", clock, {})

        def read_file(filename):
            os.makedirs(WORKSPACE_DIR, exist_ok=True)
            ruta = os.path.abspath(os.path.join(WORKSPACE_DIR, filename))
            if not ruta.startswith(WORKSPACE_DIR):
                return {"error": "Acceso denegado fuera del workspace."}
            if not os.path.exists(ruta):
                return {"error": "El archivo no existe."}
            with open(ruta, "r", encoding="utf-8") as f:
                return {"content": f.read()}
        self.registrar("read_file", "Lee archivo en workspace", read_file, {"filename": str})

        def write_file(filename, content):
            os.makedirs(WORKSPACE_DIR, exist_ok=True)
            ruta = os.path.abspath(os.path.join(WORKSPACE_DIR, filename))
            if not ruta.startswith(WORKSPACE_DIR):
                return {"error": "Acceso denegado fuera del workspace."}
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "message": f"Escrito {filename}"}
        self.registrar("write_file", "Escribe archivo en workspace", write_file,
                        {"filename": str, "content": str})

    def ejecutar(self, nombre, args):
        if nombre not in self.tools:
            return {"error": f"Herramienta '{nombre}' no encontrada."}
        tool = self.tools[nombre]
        for param, tipo in tool["schema"].items():
            if param not in args or not isinstance(args[param], tipo):
                return {"error": f"Parametro requerido faltante o tipo incorrecto: '{param}'"}
        try:
            return tool["func"](**args)
        except Exception as e:
            return {"error": str(e)}
