import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmarks.benchmark_engine import BenchmarkEngine

if __name__ == "__main__":
    print("--- BENCHMARK Y DIAGNOSTICO ---")
    engine = BenchmarkEngine()
    res = engine.ejecutar_benchmark_completo()

    print("\n========================================")
    print("INFORME TECNICO DE EVALUACION Y DIAGNOSTICO")
    print("========================================")
    print(f"Hardware objetivo  : {res['hardware']}")
    print(f"Parametros totales : {res['parametros']}")
    print(f"Uso de RAM         : {res['ram_mb']} MB")
    print(f"Uso de CPU         : {res['cpu_percent']} %")
    print(f"Tokens por segundo : {res['tokens_por_sec']} tok/s")
    print(f"Tiempo generacion  : {res['tiempo_generacion_s']} s")
    print(f"Perdida (loss)     : {res['loss']}")
    print(f"Texto generado     : '{res['texto_generado']}'")
    print("========================================")
    print("[OK] Diagnostico completado sin simulaciones ni APIs externas.")
