# -*- coding: utf-8 -*-
"""Pruebas de integridad end-to-end del sistema completo: tokenizador,
transformer (forward/backward/generacion), memoria, agente con
herramientas controladas, benchmark, y el ciclo de entrenamiento/evolucion.
No usa un framework de testing externo (solo `assert`), para no anadir
dependencias -- se ejecuta con `python3 tests/tests_final.py`."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.transformer.optimized_llm import OptimizedTransformerLLM
from memory.advanced_memory import AdvancedMemoryManager
from agent.controlled_agent import ControlledAgent
from agent.tools.registry import ToolRegistry
from benchmarks.benchmark_engine import BenchmarkEngine
import config


def test_tokenizer_embeddings_transformer_generacion():
    modelo = OptimizedTransformerLLM()
    ids = modelo.tokenizer.codificar("hola mundo")
    assert len(ids) > 0, "Tokenizer no genero ids"

    logits = modelo.forward_rapido(ids)
    assert logits.shape == (len(ids), modelo.max_vocab_size), "Forma de logits incorrecta"

    gen = modelo.generar("hola", max_nuevos=3)
    assert isinstance(gen, str) and len(gen) > 0, "Generacion invalida"

    gen_topk = modelo.generar("hola", max_nuevos=3, top_k=5)
    assert isinstance(gen_topk, str), "Generacion con top_k invalida"
    print("[OK] Tokenizer, embeddings, transformer y generacion (con y sin top_k).")


def test_backward_reduce_perdida():
    """Verifica que el backward es REAL: tras varios pasos de entrenamiento
    sobre la MISMA secuencia, la perdida debe bajar de forma consistente
    (si el gradiente estuviera mal, esto no ocurriria de forma fiable)."""
    modelo = OptimizedTransformerLLM(seed=1)
    ids = modelo.tokenizer.codificar("sistema optimizado asus cpu local eficiente")
    entrada, objetivo = ids[:-1], ids[1:]

    perdidas = []
    for _ in range(30):
        loss = modelo.paso_entrenamiento(entrada, objetivo, lr=0.01, optimizador="adam")
        perdidas.append(loss)

    assert perdidas[-1] < perdidas[0], (
        f"La perdida no bajo tras 30 pasos de entrenamiento: {perdidas[0]:.4f} -> {perdidas[-1]:.4f}"
    )
    print(f"[OK] Backward real: perdida {perdidas[0]:.4f} -> {perdidas[-1]:.4f} tras 30 pasos.")


def test_persistencia_modelo():
    """El state_dict/load_state_dict debe reproducir EXACTAMENTE la misma
    salida (roundtrip), incluyendo el estado del optimizador Adam."""
    modelo = OptimizedTransformerLLM(seed=2)
    ids = modelo.tokenizer.codificar("hola mundo")
    modelo.paso_entrenamiento(ids[:-1], ids[1:], lr=0.01, optimizador="adam")

    estado = modelo.state_dict()

    modelo2 = OptimizedTransformerLLM(seed=999)  # semilla distinta a proposito
    modelo2.load_state_dict(estado)

    logits1 = modelo.forward_rapido(ids)
    logits2 = modelo2.forward_rapido(ids)
    assert (logits1 == logits2).all(), "El roundtrip de state_dict no reproduce los mismos logits"
    print("[OK] Persistencia (state_dict/load_state_dict) reproduce el modelo exactamente.")


def test_memoria_avanzada():
    test_mem = os.path.join(os.path.dirname(__file__), "test_mem_tmp.json")
    if os.path.exists(test_mem):
        os.remove(test_mem)
    mem = AdvancedMemoryManager(filepath=test_mem)
    mem.guardar_largo_plazo("test_key", "valor_test")
    assert mem.recuperar_largo_plazo("test_key") == "valor_test"
    assert mem.actualizar_largo_plazo("test_key", "valor_actualizado")
    assert mem.recuperar_largo_plazo("test_key") == "valor_actualizado"
    assert mem.eliminar_largo_plazo("test_key")
    assert mem.recuperar_largo_plazo("test_key") is None
    if os.path.exists(test_mem):
        os.remove(test_mem)
    print("[OK] Memoria avanzada (guardar/actualizar/eliminar).")


def test_agente_y_herramientas_seguras():
    agente = ControlledAgent()
    res = agente.ejecutar_plan_autonomo([{"tool": "calculator", "arguments": {"expression": "10 + 5"}}])
    assert res[0]["resultado"]["result"] == 15
    print("[OK] Agente y herramientas controladas.")


def test_calculadora_rechaza_codigo_arbitrario():
    """La calculadora se parsea con `ast`, no `eval()` -- debe rechazar
    cualquier cosa que no sea una expresion aritmetica pura."""
    registry = ToolRegistry()
    intentos_maliciosos = [
        "__import__('os').system('echo hackeado')",
        "open('/etc/passwd').read()",
        "[].__class__.__mro__[1].__subclasses__()",
    ]
    for expr in intentos_maliciosos:
        resultado = registry.ejecutar("calculator", {"expression": expr})
        assert "error" in resultado, f"La calculadora NO rechazo una expresion peligrosa: {expr!r}"
    print("[OK] La calculadora rechaza codigo Python arbitrario (no usa eval()).")


def test_workspace_bloquea_path_traversal():
    """read_file/write_file deben rechazar rutas que intenten salir del
    WORKSPACE_DIR configurado (p.ej. ../../etc/passwd)."""
    registry = ToolRegistry()
    resultado = registry.ejecutar("read_file", {"filename": "../../../etc/passwd"})
    assert "error" in resultado, "read_file no bloqueo un intento de path traversal"
    print("[OK] El workspace del agente bloquea intentos de salir de su directorio.")


def test_benchmark():
    engine = BenchmarkEngine()
    bm = engine.ejecutar_benchmark_completo()
    assert bm["parametros"] > 0
    print(f"[OK] Benchmark real local. Parametros: {bm['parametros']}")


def main():
    print("--- EJECUTANDO PRUEBAS DE INTEGRIDAD ---\n")
    test_tokenizer_embeddings_transformer_generacion()
    test_backward_reduce_perdida()
    test_persistencia_modelo()
    test_memoria_avanzada()
    test_agente_y_herramientas_seguras()
    test_calculadora_rechaza_codigo_arbitrario()
    test_workspace_bloquea_path_traversal()
    test_benchmark()
    print("\n--- TODAS LAS PRUEBAS SUPERADAS CON EXITO ---")


if __name__ == "__main__":
    main()
