import numpy as np
import sys
import os

# Reutilizamos el tokenizador de la etapa 14 para mantener la consistencia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../etapa_14_integracion')))
from core.language_model.llm_engine import TokenizadorUnificado

class OptimizedTransformerLLM:
    def __init__(self, max_vocab_size=250, dim=16, dim_qk=16, dim_ffn=32):
        self.max_vocab_size = max_vocab_size
        self.dim = dim
        self.tokenizer = TokenizadorUnificado(max_vocab_size)
        
        np.random.seed(42)
        # Inicialización compacta optimizada para baja memoria en CPU
        self.embeddings = np.random.randn(max_vocab_size, dim) * 0.1
        self.W_q = np.random.randn(dim, dim_qk) * 0.1
        self.W_k = np.random.randn(dim, dim_qk) * 0.1
        self.W_v = np.random.randn(dim, dim) * 0.1
        self.W_out = np.random.randn(dim, max_vocab_size) * 0.1
        self.b_out = np.zeros((1, max_vocab_size))
        
        self.tokenizer.registrar_textos(["hola mundo", "sistema optimizado asus", "cpu local eficiente"])

    def softmax_vectorizado(self, x):
        # Estabilidad numérica y optimización vectorial en NumPy
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward_rapido(self, ids):
        # Búsqueda de embeddings optimizada por indexación directa de NumPy
        emb = self.embeddings[ids]
        
        # Proyecciones vectorizadas de atención
        Q = np.dot(emb, self.W_q)
        K = np.dot(emb, self.W_k)
        V = np.dot(emb, self.W_v)
        
        # Puntuaciones de atención escaladas con máscara precalculada
        scores = np.dot(Q, K.T) / np.sqrt(self.W_q.shape[1])
        T = len(ids)
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        scores += mask
        
        pesos = self.softmax_vectorizado(scores)
        contexto = np.dot(pesos, V)
        
        logits = np.dot(contexto, self.W_out) + self.b_out
        return logits

    def generar(self, prompt, max_nuevos=6):
        ids = self.tokenizer.codificar(prompt)
        if not ids or ids[0] == self.tokenizer.stoi.get("<UNK>", 1):
            ids = [self.tokenizer.stoi.get("hola", 1)]
            
        for _ in range(max_nuevos):
            logits = self.forward_rapido(ids)
            probs = self.softmax_vectorizado(logits[-1])
            siguiente = np.random.choice(len(probs), p=probs)
            ids.append(siguiente)
            
        return self.tokenizer.decodificar(ids)
