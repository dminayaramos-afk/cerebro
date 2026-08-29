# -*- coding: utf-8 -*-
"""Benchmark y diagnóstico del modelo local: parámetros reales, uso de
RAM/CPU y velocidad de generación, sin simulaciones ni llamadas externas."""

import os
import time

import numpy as np

import config
from core.transformer.optimized_llm import OptimizedTransformerLLM

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class BenchmarkEngine:
    def __init__(self, modelo=None):
        self.modelo = modelo or OptimizedTransformerLLM(
            max_vocab_size=config.MAX_VOCAB_SIZE,
            dim=config.DIM_EMBEDDING,
            dim_qk=config.DIM_QK,
            dim_ffn=config.DIM_FFN,
        )

    def contar_parametros(self):
        # Conteo exacto de parámetros de todas las matrices de pesos propias.
        m = self.modelo
        params = (
            m.embeddings.size + m.W_q.size + m.W_k.size + m.W_v.size +
            m.W1.size + m.b1.size + m.W2.size + m.b2.size +
            m.W_out.size + m.b_out.size
        )
        return int(params)

    def medir_memoria_ram(self):
        if HAS_PSUTIL:
            proceso = psutil.Process(os.getpid())
            return float(proceso.memory_info().rss / (1024 * 1024))  # MB
        return 0.0

    def medir_uso_cpu(self):
        if HAS_PSUTIL:
            return float(psutil.cpu_percent(interval=0.1))
        return 0.0

    def ejecutar_benchmark_completo(self):
        params = self.contar_parametros()

        prompt = "hola mundo"
        inicio = time.time()
        texto_gen = self.modelo.generar(prompt, max_nuevos=6, temperatura=0.8)
        tiempo_gen = time.time() - inicio

        tokens_generados = len(texto_gen.split())
        tokens_por_segundo = float(tokens_generados / max(tiempo_gen, 1e-5))

        ram_mb = self.medir_memoria_ram()
        cpu_uso = self.medir_uso_cpu()

        ids = self.modelo.tokenizer.codificar(prompt)
        if len(ids) >= 2:
            logits = self.modelo.forward_rapido(ids[:-1])
            loss_estimada, _ = self.modelo.perdida_entropia_cruzada(logits, ids[1:])
        else:
            loss_estimada = float("nan")

        return {
            "hardware": "CPU local",
            "parametros": params,
            "ram_mb": round(ram_mb, 2),
            "cpu_percent": cpu_uso,
            "tokens_por_sec": round(tokens_por_segundo, 2),
            "tiempo_generacion_s": round(tiempo_gen, 4),
            "loss": round(loss_estimada, 4) if not np.isnan(loss_estimada) else None,
            "texto_generado": texto_gen,
        }
