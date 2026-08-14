import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmarks.benchmark_engine import BenchmarkEngine

if __name__ == "__main__":
    print("--- INICIANDO BENCHMARK Y DIAGNÓSTICO (ETAPA 15) ---")
    engine = BenchmarkEngine()
    res = engine.ejecutar_benchmark_completo()
    
    print("\n========================================")
    print("📊 INFORME TÉCNICO DE EVALUACIÓN Y DIAGNÓSTICO")
    print("========================================")
    print(f"Hardware objetivo  : {res['hardware']}")
    print(f"Parámetros totales : {res['parametros']}")
    print(f"Uso de RAM         : {res['ram_mb']} MB")
    print(f"Uso de CPU         : {res['cpu_percent']} %")
    print(f"Tokens por segundo : {res['tokens_por_sec']} tok/s")
    print(f"Tiempo generación  : {res['tiempo_generacion_s']} s")
    print(f"Pérdida (Loss)     : {res['loss']}")
    print(f"Texto generado     : '{res['texto_generado']}'")
    print("========================================")
    print("[✔] Diagnóstico completado sin simulaciones ni APIs externas.")
