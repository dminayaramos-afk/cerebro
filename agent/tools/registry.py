import os
import datetime
from config import WORKSPACE_DIR

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.registrar_base()

    def registrar(self, nombre, desc, func, schema):
        self.tools[nombre] = {"desc": desc, "func": func, "schema": schema}

    def registrar_base(self):
        def calc(expression):
            if not all(c in "0123456789+-*/(). " for c in expression):
                return {"error": "Expresión inválida o caracteres no permitidos."}
            try:
                return {"result": eval(expression, {"__builtins__": None}, {})}
            except Exception as e:
                return {"error": str(e)}
        self.registrar("calculator", "Calculadora matemática segura", calc, {"expression": str})

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
            with open(ruta, 'r', encoding='utf-8') as f:
                return {"content": f.read()}
        self.registrar("read_file", "Lee archivo en workspace", read_file, {"filename": str})

        def write_file(filename, content):
            os.makedirs(WORKSPACE_DIR, exist_ok=True)
            ruta = os.path.abspath(os.path.join(WORKSPACE_DIR, filename))
            if not ruta.startswith(WORKSPACE_DIR):
                return {"error": "Acceso denegado fuera del workspace."}
            with open(ruta, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"status": "success", "message": f"Escrito {filename}"}
        self.registrar("write_file", "Escribe archivo en workspace", write_file, {"filename": str, "content": str})

    def ejecutar(self, nombre, args):
        if nombre not in self.tools:
            return {"error": f"Herramienta '{nombre}' no encontrada."}
        tool = self.tools[nombre]
        for param, tipo in tool["schema"].items():
            if param not in args or not isinstance(args[param], tipo):
                return {"error": f"Parámetro requerido faltante o tipo incorrecto: '{param}'"}
        try:
            return tool["func"](**args)
        except Exception as e:
            return {"error": str(e)}
