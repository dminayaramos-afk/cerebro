import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.transformer.optimized_llm import OptimizedTransformerLLM
from memory.advanced_memory import AdvancedMemoryManager
from agent.controlled_agent import ControlledAgent
from benchmarks.benchmark_engine import BenchmarkEngine

def mostrar_status(modelo, memoria):
    print("\n=== ESTADO DEL CEREBRO IA (v1.0) ===")
    print(f"Modelo: Transformer Optimizado local")
    print(f"Vocabulario: {len(modelo.tokenizer.stoi)} tokens")
    print(f"Memoria Largo Plazo: {len(memoria.long_term_memory)} elementos")
    print(f"Hardware: ASUS X550LC (CPU)")
    print("=====================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MI CEREBRO IA v1.0 - Sistema Central Local")
    parser.add_argument("--debug", action="store_true", help="Activar modo debug")
    args = parser.parse_args()

    modelo = OptimizedTransformerLLM()
    memoria = AdvancedMemoryManager(filepath=os.path.join(os.path.dirname(__file__), "memory/memory_store.json"))
    agente = ControlledAgent(max_steps=4, max_tool_calls=3)

    print("\n========================================")
    print("🧠 MI CEREBRO IA v1.0 [ESTABLE LOCAL]")
    print("Escribe '/help' para comandos o interactúa.")
    print("========================================\n")

    while True:
        try:
            user_input = input("Usuario: ").strip()
            if not user_input:
                continue
            
            if user_input == "/exit":
                print("IA: ¡Hasta pronto!")
                break
            elif user_input == "/help":
                print("Comandos: /help, /status, /memory, /exit")
                continue
            elif user_input == "/status":
                mostrar_status(modelo, memoria)
                continue
            elif user_input == "/memory":
                print(f"Memoria de largo plazo: {memoria.long_term_memory}")
                continue

            # Registro de contexto de corto plazo
            memoria.agregar_contexto("usuario", user_input)
            
            # Generación de respuesta con nuestro propio modelo
            respuesta = modelo.generar(user_input, max_nuevos=6)
            print(f"IA: {respuesta}")
            
            memoria.agregar_contexto("ia", respuesta)

        except (KeyboardInterrupt, EOFError):
            print("\nIA: Saliendo...")
            break
