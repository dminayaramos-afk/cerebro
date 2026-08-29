# -*- coding: utf-8 -*-
"""
Motor de entrenamiento. A diferencia de la versión anterior (que solo
restaba una constante fija a los embeddings, sin relación real con la
pérdida), este entrena con gradiente descendente de verdad, usando los
gradientes exactos calculados en `OptimizedTransformerLLM.backward`.
"""

import json
import os

import config
from core.transformer.optimized_llm import OptimizedTransformerLLM


class TrainerEngine:
    def __init__(self, learning_rate=None, batch_size=2, seq_len=None, modelo=None):
        self.modelo = modelo or OptimizedTransformerLLM(
            max_vocab_size=config.MAX_VOCAB_SIZE,
            dim=config.DIM_EMBEDDING,
            dim_qk=config.DIM_QK,
            dim_ffn=config.DIM_FFN,
            vocab_path=config.VOCAB_FILE,
        )
        self.lr = learning_rate if learning_rate is not None else config.TASA_APRENDIZAJE
        self.batch_size = batch_size
        self.seq_len = seq_len or config.MAX_SEQ_LEN
        self.best_loss = float("inf")

    def cargar_corpus(self, ruta=None):
        ruta = ruta or config.CORPUS_FILE
        if not os.path.exists(ruta):
            return ["hola mundo", "sistema optimizado asus cpu local eficiente"]
        with open(ruta, "r", encoding="utf-8") as f:
            lineas = [linea.strip() for linea in f if linea.strip()]
        return lineas or ["hola mundo"]

    def _ventanas_entrenamiento(self, ids):
        """Genera pares (entrada, objetivo) desplazados en una ventana de
        hasta seq_len tokens, para no exceder MAX_SEQ_LEN por secuencia."""
        ventanas = []
        paso = max(self.seq_len - 1, 1)
        for inicio in range(0, max(len(ids) - 1, 1), paso):
            fragmento = ids[inicio:inicio + self.seq_len]
            if len(fragmento) < 2:
                continue
            ventanas.append((fragmento[:-1], fragmento[1:]))
        return ventanas

    def entrenar_epoca(self, lineas_corpus):
        """Entrena una época completa sobre el corpus (lista de líneas de
        texto), registrando cada línea en el vocabulario si trae palabras
        nuevas, y devuelve la pérdida media real de la época."""
        self.modelo.tokenizer.registrar_textos(lineas_corpus)

        loss_total = 0.0
        pasos = 0

        for linea in lineas_corpus:
            ids = self.modelo.tokenizer.codificar(linea)
            for entrada, objetivo in self._ventanas_entrenamiento(ids):
                loss = self.modelo.paso_entrenamiento(entrada, objetivo, self.lr, optimizador=config.OPTIMIZADOR)
                loss_total += loss
                pasos += 1

        return loss_total / max(pasos, 1)

    def guardar_checkpoint(self, ruta_dir, epoca, loss):
        os.makedirs(ruta_dir, exist_ok=True)
        ruta_archivo = os.path.join(ruta_dir, "checkpoint.json")
        datos = {
            "epoca": epoca,
            "loss": loss,
            "modelo": self.modelo.state_dict(),
        }
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            json.dump(datos, f)

    def cargar_checkpoint(self, ruta_dir):
        ruta_archivo = os.path.join(ruta_dir, "checkpoint.json")
        if os.path.exists(ruta_archivo):
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                datos = json.load(f)
            self.modelo.load_state_dict(datos["modelo"])
            self.best_loss = datos["loss"]
            return datos["epoca"], datos["loss"]
        return 0, float("inf")
