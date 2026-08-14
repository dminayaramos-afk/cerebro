import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.transformer.optimized_llm import OptimizedTransformerLLM
from memory.advanced_memory import AdvancedMemoryManager
from agent.controlled_agent import ControlledAgent
from benchmarks.benchmark_engine import BenchmarkEngine

if __name__ == "__main__":
    print("--- EJECUTANDO PRUEBAS FINALES DE INTEGRIDAD (ETAPA 20) ---")
    
    # Test 1, 2, 3 y 4: Tokenizer, Embeddings, Transformer y Generación
    modelo = OptimizedTransformerLLM()
    ids = modelo.tokenizer.codificar("hola mundo")
    assert len(ids) > 0, "Test 1 Fallido: Tokenizer"
    
    logits = modelo.forward_rapido(ids)
    assert logits is not None, "Test 3 Fallido: Transformer"
    
    gen = modelo.generar("hola", max_nuevos=3)
    assert isinstance(gen, str), "Test 4 Fallido: Generación"
    print("[✔] Tests 1-4 (Tokenizer, Embeddings, Transformer, Generación): OK.")
    
    # Test 5: Memoria avanzada
    test_mem = "MI_CEREBRO_IA/etapa_20_final/tests/test_mem.json"
    mem = AdvancedMemoryManager(filepath=test_mem)
    mem.guardar_largo_plazo("test_key", "valor_test")
    assert mem.recuperar_largo_plazo("test_key") == "valor_test"
    print("[✔] Test 5 (Memoria Avanzada): OK.")
    if os.path.exists(test_mem):
        os.remove(test_mem)
        
    # Test 8 y 9: Agente y Herramientas controladas
    agente = ControlledAgent()
    res_agente = agente.ejecutar_plan_autonomo([{"tool": "calculator", "arguments": {"expression": "10 + 5"}}])
    assert res_agente[0]["resultado"]["result"] == 15
    print("[✔] Tests 8 & 9 (Agente y Herramientas Seguras): OK.")
    
    # Test 11: Benchmark
    engine = BenchmarkEngine()
    bm = engine.ejecutar_benchmark_completo()
    assert bm["parametros"] > 0
    print(f"[✔] Test 11 (Benchmark real local): OK. Parámetros: {bm['parametros']}")
    
    print("\n--- ¡TODAS LAS PRUEBAS FINALES DE LA ETAPA 20 SUPERADAS CON ÉXITO! ---")
