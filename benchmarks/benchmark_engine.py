import time
import os
import sys
import numpy as np

# Añadir la ruta de la etapa 14/core para reutilizar nuestro modelo integrado
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../etapa_14_integracion')))
from core.language_model.llm_engine import TransformerLLM

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class BenchmarkEngine:
    def __init__(self):
        self.modelo = TransformerLLM()

    def contar_parametros(self):
        # Conteo exacto de parámetros de nuestras matrices de pesos propias
        params = (
            self.modelo.embeddings.matriz.size +
            self.modelo.W_q.size +
            self.modelo.W_k.size +
            self.modelo.W_v.size +
            self.modelo.W_out.size +
            self.modelo.b_out.size
        )
        return int(params)

    def medir_memoria_ram(self):
        if HAS_PSUTIL:
            proceso = psutil.Process(os.getpid())
            return float(proceso.memory_info().rss / (1024 * 1024)) # En MB
        return 0.0

    def medir_uso_cpu(self):
        if HAS_PSUTIL:
            return float(psutil.cpu_percent(interval=0.1))
        return 0.0

    def ejecutar_benchmark_completo(self):
        params = self.contar_parametros()
        ram_inicial = self.medir_memoria_ram()
        
        # Medir velocidad de generación y tokens por segundo
        prompt = "hola mundo"
        inicio = time.time()
        texto_gen = self.modelo.generar(prompt, max_nuevos=6, temperatura=0.8)
        tiempo_gen = time.time() - inicio
        
        tokens_generados = len(texto_gen.split())
        tokens_por_segundo = float(tokens_generados / max(tiempo_gen, 1e-5))
        
        ram_final = self.medir_memoria_ram()
        cpu_uso = self.medir_uso_cpu()
        
        # Pérdida estimada inicial sobre corpus base
        ids = self.modelo.tokenizer.codificar(prompt)
        logits = self.modelo.forward(ids)
        probs = self.modelo.softmax(logits[-1])
        loss_estimada = float(-np.log(max(probs[ids[0] if ids else 0], 1e-15)))

        resultados = {
            "hardware": "ASUS X550LC (CPU Local)",
            "parametros": params,
            "ram_mb": round(ram_final, 2),
            "cpu_percent": cpu_uso,
            "tokens_por_sec": round(tokens_por_segundo, 2),
            "tiempo_generacion_s": round(tiempo_gen, 4),
            "loss": round(loss_estimada, 4),
            "texto_generado": texto_gen
        }
        return resultados
