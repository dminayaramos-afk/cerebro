import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from core.transformer.optimized_llm import OptimizedTransformerLLM
from memory.advanced_memory import AdvancedMemoryManager
from agent.controlled_agent import ControlledAgent


def mostrar_status(modelo, memoria):
    print("\n=== ESTADO DEL CEREBRO IA ===")
    print("Modelo: Transformer minimalista local (NumPy, desde cero)")
    print(f"Vocabulario: {len(modelo.tokenizer)} tokens")
    print(f"Memoria largo plazo: {len(memoria.long_term_memory)} elementos")
    print(f"Parametros del modelo: {sum(a.size for a in (modelo.embeddings, modelo.W_q, modelo.W_k, modelo.W_v, modelo.W1, modelo.b1, modelo.W2, modelo.b2, modelo.W_out, modelo.b_out))}")
    print("==============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mi Cerebro IA - Sistema central local")
    parser.add_argument("--debug", action="store_true", help="Activar modo debug")
    args = parser.parse_args()

    modelo = OptimizedTransformerLLM(
        max_vocab_size=config.MAX_VOCAB_SIZE,
        dim=config.DIM_EMBEDDING,
        dim_qk=config.DIM_QK,
        dim_ffn=config.DIM_FFN,
        vocab_path=config.VOCAB_FILE,
    )

    # Si hay un checkpoint entrenado, cargarlo para usar los pesos aprendidos
    # en vez de los pesos aleatorios iniciales.
    ruta_checkpoint = os.path.join(config.DIR_CHECKPOINTS, "checkpoint.json")
    if os.path.exists(ruta_checkpoint):
        import json
        with open(ruta_checkpoint, "r", encoding="utf-8") as f:
            datos = json.load(f)
        modelo.load_state_dict(datos["modelo"])
        if args.debug:
            print(f"[debug] Checkpoint cargado (epoca {datos['epoca']}, loss {datos['loss']:.4f})")

    memoria = AdvancedMemoryManager()
    agente = ControlledAgent()

    print("\n========================================")
    print("MI CEREBRO IA [local]")
    print("Escribe '/help' para comandos o interactua.")
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

            memoria.agregar_contexto("usuario", user_input)

            respuesta = modelo.generar(user_input, max_nuevos=6, temperatura=0.8, top_k=10)
            print(f"IA: {respuesta}")

            memoria.agregar_contexto("ia", respuesta)

        except (KeyboardInterrupt, EOFError):
            print("\nIA: Saliendo...")
            break
