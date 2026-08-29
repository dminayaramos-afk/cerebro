import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from trainer_engine import TrainerEngine


def dividir_train_val(lineas, proporcion_val=0.2):
    """División simple train/val (sin mezclar aleatoriamente, para que el
    experimento sea reproducible entre corridas)."""
    n_val = max(1, int(len(lineas) * proporcion_val)) if len(lineas) > 1 else 0
    if n_val == 0:
        return lineas, lineas  # corpus muy pequeño: valida sobre lo mismo
    return lineas[:-n_val], lineas[-n_val:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de Mi Cerebro IA")
    parser.add_argument("--resume", action="store_true", help="Continuar desde el último checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento")
    parser.add_argument("--corpus", default=None, help="Ruta a un archivo de corpus alternativo (una frase por línea)")
    args = parser.parse_args()

    trainer = TrainerEngine()

    epoca_inicial = 0
    if args.resume:
        epoca_inicial, loss_prev = trainer.cargar_checkpoint(config.DIR_CHECKPOINTS)
        if epoca_inicial > 0:
            print(f"[OK] Reanudando entrenamiento desde epoca {epoca_inicial} (loss anterior: {loss_prev:.4f})")
        else:
            print("[i] No se encontro checkpoint previo, empezando desde cero.")

    lineas = trainer.cargar_corpus(args.corpus)
    lineas_train, lineas_val = dividir_train_val(lineas)
    print(f"Corpus: {len(lineas)} lineas totales ({len(lineas_train)} entrenamiento / {len(lineas_val)} validacion)")

    print("\n--- INICIANDO ENTRENAMIENTO ---")
    for epoca in range(epoca_inicial + 1, epoca_inicial + args.epochs + 1):
        loss_train = trainer.entrenar_epoca(lineas_train)

        # Validación real: mide la pérdida sobre frases no vistas en esta
        # época, sin actualizar los pesos (no es una cifra inventada).
        loss_val_total, pasos_val = 0.0, 0
        for linea in lineas_val:
            ids = trainer.modelo.tokenizer.codificar(linea)
            for entrada, objetivo in trainer._ventanas_entrenamiento(ids):
                logits = trainer.modelo.forward_rapido(entrada)
                loss_i, _ = trainer.modelo.perdida_entropia_cruzada(logits, objetivo)
                loss_val_total += loss_i
                pasos_val += 1
        loss_val = loss_val_total / max(pasos_val, 1)

        print(f"Epoca: {epoca} | Loss entrenamiento: {loss_train:.4f} | "
              f"Loss validacion: {loss_val:.4f} | Learning rate: {trainer.lr}")

        if loss_val < trainer.best_loss:
            trainer.best_loss = loss_val
            trainer.guardar_checkpoint(config.DIR_CHECKPOINTS, epoca, loss_val)
            print(f"[OK] Nuevo mejor checkpoint guardado en epoca {epoca} (loss val: {loss_val:.4f})")

    print("--- ENTRENAMIENTO COMPLETADO ---")
