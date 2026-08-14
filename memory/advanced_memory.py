import os
import json
import datetime

class AdvancedMemoryManager:
    def __init__(self, filepath="MI_CEREBRO_IA/etapa_18_memoria_avanzada/memory_store.json"):
        self.filepath = filepath
        self.short_term_context = []  # Contexto activo de corto plazo
        self.long_term_memory = {}    # Memoria de largo plazo con metadatos
        self.cargar_memoria()

    def agregar_contexto(self, rol, contenido):
        # Mantiene un buffer de corto plazo limpio
        self.short_term_context.append({"rol": rol, "contenido": contenido})
        if len(self.short_term_context) > 5:
            self.short_term_context.pop(0)

    def guardar_largo_plazo(self, clave, valor, categoria="general", importancia=1):
        self.long_term_memory[clave] = {
            "valor": valor,
            "categoria": categoria,
            "importancia": importancia,
            "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.persistir()

    def recuperar_largo_plazo(self, clave):
        if clave in self.long_term_memory:
            return self.long_term_memory[clave]["valor"]
        return None

    def actualizar_largo_plazo(self, clave, nuevo_valor):
        if clave in self.long_term_memory:
            self.long_term_memory[clave]["valor"] = nuevo_valor
            self.long_term_memory[clave]["fecha"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.persistir()
            return True
        return False

    def eliminar_largo_plazo(self, clave):
        if clave in self.long_term_memory:
            del self.long_term_memory[clave]
            self.persistir()
            return True
        return False

    def persistir(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        datos = {
            "long_term": self.long_term_memory
        }
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)

    def cargar_memoria(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                    self.long_term_memory = datos.get("long_term", {})
            except Exception:
                self.long_term_memory = {}
