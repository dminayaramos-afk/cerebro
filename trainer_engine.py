import numpy as np
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../etapa_16_optimizacion')))
from core.transformer.optimized_llm import OptimizedTransformerLLM

class TrainerEngine:
    def __init__(self, learning_rate=0.01, batch_size=2):
        self.modelo = OptimizedTransformerLLM()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.best_loss = float('inf')

    def calcular_perdida(self, logits, targets):
        # Pérdida de entropía cruzada simplificada
        T, V = logits.shape
        loss = 0.0
        for t in range(T):
            exp_l = np.exp(logits[t] - np.max(logits[t]))
            probs = exp_l / np.sum(exp_l)
            true_p = max(probs[targets[t]], 1e-15)
            loss -= np.log(true_p)
        return float(loss / T)

    def entrenar_epoca(self, corpus_ids):
        loss_total = 0.0
        pasos = 0
        
        # Simulación de mini-batches por secuencias
        for i in range(0, len(corpus_ids) - 1, self.batch_size):
            batch = corpus_ids[i:i + self.batch_size + 1]
            if len(batch) < 2:
                continue
                
            inputs = batch[:-1]
            targets = batch[1:]
            
            logits = self.modelo.forward_rapido(inputs)
            loss = self.calcular_perdida(logits, targets)
            
            # Actualización simulada mediante gradientes estocásticos propios
            self.modelo.embeddings[inputs] -= self.lr * 0.001
            loss_total += loss
            pasos += 1
            
        loss_promedio = loss_total / max(pasos, 1)
        return loss_promedio

    def guardar_checkpoint(self, ruta_dir, epoca, loss):
        os.makedirs(ruta_dir, exist_ok=True)
        ruta_archivo = os.path.join(ruta_dir, "checkpoint_etapa17.json")
        datos = {
            "epoca": epoca,
            "loss": loss,
            "embeddings": self.modelo.embeddings.tolist()
        }
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            json.dump(datos, f)

    def cargar_checkpoint(self, ruta_dir):
        ruta_archivo = os.path.join(ruta_dir, "checkpoint_etapa17.json")
        if os.path.exists(ruta_archivo):
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                datos = json.load(f)
                self.modelo.embeddings = np.array(datos["embeddings"])
                return datos["epoca"], datos["loss"]
        return 0, float('inf')
