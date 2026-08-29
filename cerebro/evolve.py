# -*- coding: utf-8 -*-
"""
Ciclo de "evolucion" del modelo: busqueda real de hiperparametros.

No es una simulacion ni una lista de numeros inventados -- entrena varios
modelos candidatos desde cero (con distintas combinaciones de learning
rate y semilla aleatoria) durante un numero reducido de epocas cada uno,
mide la perdida de validacion real de cada candidato, y se queda con el
que mejor generaliza. Ese candidato ganador se entrena luego durante mas
epocas y su checkpoint final sustituye al checkpoint principal solo si
mejora lo que ya habia.

Uso:
    python3 evolve.py --candidatos 4 --epocas-candidato 5 --epocas-final 15
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from train import dividir_train_val
from trainer_engine import TrainerEngine


def generar_candidatos(n):
    """Combinaciones de hiperparametros a probar: variando la tasa de
    aprendizaje base de config.py y la semilla del modelo, para explorar
    tanto la velocidad de aprendizaje como la sensibilidad a la
    inicializacion aleatoria (relevante en un modelo tan pequeno)."""
    lrs = [config.TASA_APRENDIZAJE * factor for factor in (0.5, 1.0, 2.0, 4.0)]
    semillas = [42, 7, 123, 2024]
    candidatos = []
    for i in range(n):
        candidatos.append({
            "lr": lrs[i % len(lrs)],
            "seed": semillas[i % len(semillas)],
        })
    return candidatos


def evaluar_candidato(hparams, lineas_train, lineas_val, epocas):
    from core.transformer.optimized_llm import OptimizedTransformerLLM

    modelo = OptimizedTransformerLLM(
        max_vocab_size=config.MAX_VOCAB_SIZE,
        dim=config.DIM_EMBEDDING,
        dim_qk=config.DIM_QK,
        dim_ffn=config.DIM_FFN,
        seed=hparams["seed"],
    )
    trainer = TrainerEngine(learning_rate=hparams["lr"], modelo=modelo)

    for _ in range(epocas):
        trainer.entrenar_epoca(lineas_train)

    loss_val_total, pasos_val = 0.0, 0
    for linea in lineas_val:
        ids = trainer.modelo.tokenizer.codificar(linea)
        for entrada, objetivo in trainer._ventanas_entrenamiento(ids):
            logits = trainer.modelo.forward_rapido(entrada)
            loss_i, _ = trainer.modelo.perdida_entropia_cruzada(logits, objetivo)
            loss_val_total += loss_i
            pasos_val += 1
    loss_val = loss_val_total / max(pasos_val, 1)

    return loss_val, trainer


def main():
    parser = argparse.ArgumentParser(
        description="Busqueda de hiperparametros (evolucion) para Mi Cerebro IA")
    parser.add_argument("--candidatos", type=int, default=4,
                         help="Numero de combinaciones de hiperparametros a probar.")
    parser.add_argument("--epocas-candidato", type=int, default=5,
                         help="Epocas de entrenamiento rapido por candidato.")
    parser.add_argument("--epocas-final", type=int, default=15,
                         help="Epocas de entrenamiento completo para el candidato ganador.")
    parser.add_argument("--corpus", default=None,
                         help="Ruta a un corpus alternativo (una frase por linea).")
    args = parser.parse_args()

    trainer_base = TrainerEngine()
    lineas = trainer_base.cargar_corpus(args.corpus)
    lineas_train, lineas_val = dividir_train_val(lineas)
    print(f"Corpus: {len(lineas)} lineas ({len(lineas_train)} entrenamiento / {len(lineas_val)} validacion)")

    candidatos = generar_candidatos(args.candidatos)

    print(f"\n--- EVALUANDO {len(candidatos)} CANDIDATOS ({args.epocas_candidato} epocas c/u) ---")
    resultados = []
    for i, hparams in enumerate(candidatos, start=1):
        loss_val, _ = evaluar_candidato(hparams, lineas_train, lineas_val, args.epocas_candidato)
        print(f"  Candidato {i}: lr={hparams['lr']:.4f} seed={hparams['seed']} "
              f"-> loss_val={loss_val:.4f}")
        resultados.append((loss_val, hparams))

    resultados.sort(key=lambda r: r[0])
    mejor_loss, mejor_hparams = resultados[0]
    print(f"\n[OK] Mejor candidato: lr={mejor_hparams['lr']:.4f} seed={mejor_hparams['seed']} "
          f"(loss_val preliminar: {mejor_loss:.4f})")

    print(f"\n--- ENTRENAMIENTO COMPLETO DEL GANADOR ({args.epocas_final} epocas) ---")
    loss_final, trainer_final = evaluar_candidato(
        mejor_hparams, lineas_train, lineas_val, args.epocas_final)
    print(f"Loss de validacion final: {loss_final:.4f}")

    ruta_principal = os.path.join(config.DIR_CHECKPOINTS, "checkpoint.json")
    if os.path.exists(ruta_principal):
        # Se lee solo el valor de loss del checkpoint existente para
        # comparar -- NO se usa trainer_final.cargar_checkpoint() aqui,
        # porque eso sobreescribiria los pesos recien entrenados del
        # candidato ganador con los del checkpoint viejo.
        import json
        with open(ruta_principal, "r", encoding="utf-8") as f:
            loss_actual = json.load(f)["loss"]
    else:
        loss_actual = float("inf")

    os.makedirs(config.DIR_EVOLUTION, exist_ok=True)
    trainer_final.guardar_checkpoint(config.DIR_EVOLUTION, args.epocas_final, loss_final)
    print(f"[OK] Candidato evolucionado guardado en {config.DIR_EVOLUTION}/checkpoint.json")

    # Solo se sustituye el checkpoint principal si el candidato evolucionado
    # generaliza mejor que lo que ya habia -- comparacion honesta, no un
    # reemplazo incondicional.
    if loss_final < loss_actual:
        trainer_final.guardar_checkpoint(config.DIR_CHECKPOINTS, args.epocas_final, loss_final)
        print(f"[OK] Este candidato mejora el checkpoint principal "
              f"({loss_final:.4f} < {loss_actual:.4f}) -> sustituido.")
    else:
        print(f"[i] El checkpoint principal actual ya es mejor o igual "
              f"({loss_actual:.4f} <= {loss_final:.4f}) -> se conserva sin cambios.")


if __name__ == "__main__":
    main()
