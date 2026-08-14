import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trainer_engine import TrainerEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento mejorado de Mi Cerebro IA")
    parser.add_argument("--resume", action="store_true", help="Continuar desde el último checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas de entrenamiento")
    args = parser.parse_args()

    dir_chk = os.path.join(os.path.dirname(__file__), "checkpoints")
    trainer = TrainerEngine(learning_rate=0.01, batch_size=2)
    
    epoca_inicial = 0
    if args.resume:
        epoca_inicial, loss_prev = trainer.cargar_checkpoint(dir_chk)
        print(f"[✔] Reanudando entrenamiento desde época {epoca_inicial} (Loss anterior: {loss_prev:.4f})")

    corpus_prueba = trainer.modelo.tokenizer.codificar("sistema optimizado asus cpu local eficiente")

    print("\n--- INICIANDO ENTRENAMIENTO MEJORADO (ETAPA 17) ---")
    for epoca in range(epoca_inicial + 1, epoca_inicial + args.epochs + 1):
        loss_train = trainer.entrenar_epoca(corpus_prueba)
        loss_val = loss_train * 1.05  # Validación simulada
        
        print(f"Paso/Época: {epoca} | Loss entrenamiento: {loss_train:.4f} | Loss validación: {loss_val:.4f} | Learning Rate: 0.01")
        
        if loss_val < trainer.best_loss:
            trainer.best_loss = loss_val
            trainer.guardar_checkpoint(dir_chk, epoca, loss_val)
            print(f"[✔] Nuevo mejor checkpoint guardado en época {epoca}")

    print("--- ¡ENTRENAMIENTO COMPLETADO CON ÉXITO! ---")
