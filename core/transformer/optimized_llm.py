# -*- coding: utf-8 -*-
"""
Transformer minimalista construido desde cero con NumPy puro (sin PyTorch,
TensorFlow ni ningún framework de autodiferenciación). Incluye:

  - Embeddings entrenables.
  - Auto-atención de una sola cabeza, con máscara causal.
  - Un bloque feed-forward (FFN) con ReLU, con conexiones residuales.
  - Retropropagación manual (backward) de TODOS los parámetros, derivada
    a mano con la regla de la cadena -- no hay "entrenamiento simulado":
    los gradientes calculados aquí son los gradientes reales de la
    pérdida de entropía cruzada respecto a cada matriz de pesos.

Es intencionalmente pequeño (dimensiones configurables en config.py) para
poder entrenarse en CPU con RAM limitada. No es competitivo con un LLM
real -- es un proyecto educativo para entender, implementar y verificar de
primera mano cómo funciona un transformer por dentro.
"""

import numpy as np

from core.tokenizer import TokenizadorUnificado


class OptimizedTransformerLLM:
    def __init__(self, max_vocab_size=250, dim=16, dim_qk=16, dim_ffn=32,
                 vocab_path=None, seed=42):
        self.max_vocab_size = max_vocab_size
        self.dim = dim
        self.dim_qk = dim_qk
        self.dim_ffn = dim_ffn

        self.tokenizer = TokenizadorUnificado(max_vocab_size, vocab_path=vocab_path)

        rng = np.random.default_rng(seed)
        escala = 0.1
        # Inicialización compacta para bajo uso de memoria en CPU.
        self.embeddings = rng.standard_normal((max_vocab_size, dim)) * escala
        self.W_q = rng.standard_normal((dim, dim_qk)) * escala
        self.W_k = rng.standard_normal((dim, dim_qk)) * escala
        self.W_v = rng.standard_normal((dim, dim)) * escala
        self.W1 = rng.standard_normal((dim, dim_ffn)) * escala
        self.b1 = np.zeros((1, dim_ffn))
        self.W2 = rng.standard_normal((dim_ffn, dim)) * escala
        self.b2 = np.zeros((1, dim))
        self.W_out = rng.standard_normal((dim, max_vocab_size)) * escala
        self.b_out = np.zeros((1, max_vocab_size))

        self.tokenizer.registrar_textos([
            "hola mundo", "sistema optimizado asus", "cpu local eficiente",
        ])

        # Estado del optimizador Adam (momentos de primer y segundo orden
        # por parámetro). Se inicializa en cero y se actualiza en cada paso
        # de entrenamiento; no se usa en absoluto durante la generación.
        self._adam_m = {}
        self._adam_v = {}
        self._adam_t = 0

    # -- utilidades numéricas -------------------------------------------------

    @staticmethod
    def softmax_vectorizado(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # -- forward ---------------------------------------------------------------

    def forward_rapido(self, ids, cache=None):
        """Pasada hacia adelante. Si se pasa `cache` (un dict vacío), se
        rellena con los valores intermedios necesarios para `backward`."""
        ids = np.asarray(ids, dtype=np.int64)
        T = len(ids)

        emb = self.embeddings[ids]                       # (T, D)

        Q = emb @ self.W_q                                 # (T, Dqk)
        K = emb @ self.W_k
        V = emb @ self.W_v                                 # (T, D)

        scores = (Q @ K.T) / np.sqrt(self.dim_qk)           # (T, T)
        mask = np.triu(np.ones((T, T)), k=1) * -1e9          # causal
        scores = scores + mask

        attn = self.softmax_vectorizado(scores)              # (T, T)
        contexto = attn @ V                                  # (T, D)

        resid1 = emb + contexto                              # (T, D)

        ffn_pre = resid1 @ self.W1 + self.b1                 # (T, Dffn)
        ffn_hidden = np.maximum(ffn_pre, 0.0)                 # ReLU
        ffn_out = ffn_hidden @ self.W2 + self.b2              # (T, D)

        resid2 = resid1 + ffn_out                             # (T, D)

        logits = resid2 @ self.W_out + self.b_out              # (T, V)

        if cache is not None:
            cache.update(dict(
                ids=ids, emb=emb, Q=Q, K=K, V=V, attn=attn,
                contexto=contexto, resid1=resid1, ffn_pre=ffn_pre,
                ffn_hidden=ffn_hidden, ffn_out=ffn_out, resid2=resid2,
                logits=logits,
            ))
        return logits

    # -- pérdida -----------------------------------------------------------------

    def perdida_entropia_cruzada(self, logits, targets):
        """Cross-entropy media sobre la secuencia. Devuelve (loss, probs)."""
        probs = self.softmax_vectorizado(logits)             # (T, V)
        T = len(targets)
        p_correcta = probs[np.arange(T), targets]
        p_correcta = np.clip(p_correcta, 1e-12, 1.0)
        loss = -np.mean(np.log(p_correcta))
        return float(loss), probs

    # -- backward (retropropagación manual) -----------------------------------

    def backward(self, cache, targets):
        """Calcula los gradientes REALES de la pérdida respecto a cada
        parámetro, derivados a mano con la regla de la cadena a través de
        atención, FFN y residuales. No hay aproximaciones ni atajos: esto
        es exactamente lo que haría autograd, escrito explícitamente."""
        ids = cache["ids"]
        emb, Q, K, V = cache["emb"], cache["Q"], cache["K"], cache["V"]
        attn, resid1 = cache["attn"], cache["resid1"]
        ffn_pre, ffn_hidden, resid2 = cache["ffn_pre"], cache["ffn_hidden"], cache["resid2"]
        logits = cache["logits"]

        T = len(ids)
        targets = np.asarray(targets, dtype=np.int64)

        probs = self.softmax_vectorizado(logits)
        dlogits = probs.copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T                                         # media sobre la secuencia

        # --- capa de salida ---
        dW_out = resid2.T @ dlogits
        db_out = np.sum(dlogits, axis=0, keepdims=True)
        dresid2 = dlogits @ self.W_out.T                      # (T, D)

        # --- FFN (resid2 = resid1 + ffn_out) ---
        dffn_out = dresid2
        dW2 = ffn_hidden.T @ dffn_out
        db2 = np.sum(dffn_out, axis=0, keepdims=True)
        dffn_hidden = dffn_out @ self.W2.T
        dffn_pre = dffn_hidden * (ffn_pre > 0)                 # derivada de ReLU
        dW1 = resid1.T @ dffn_pre
        db1 = np.sum(dffn_pre, axis=0, keepdims=True)
        dresid1_via_ffn = dffn_pre @ self.W1.T

        dresid1 = dresid2 + dresid1_via_ffn                    # ambas ramas del residual

        # --- atención (resid1 = emb + contexto) ---
        dcontexto = dresid1
        dattn = dcontexto @ V.T                                # (T, T)
        dV = attn.T @ dcontexto                                # (T, D)

        # jacobiano de softmax fila a fila: dscores = attn * (dattn - sum(dattn*attn))
        suma_fila = np.sum(dattn * attn, axis=-1, keepdims=True)
        dscores = attn * (dattn - suma_fila)

        dQ = (dscores @ K) / np.sqrt(self.dim_qk)
        dK = (dscores.T @ Q) / np.sqrt(self.dim_qk)

        dW_q = emb.T @ dQ
        dW_k = emb.T @ dK
        dW_v = emb.T @ dV

        demb = dresid1 + (dQ @ self.W_q.T) + (dK @ self.W_k.T) + (dV @ self.W_v.T)

        # acumula en la matriz de embeddings completa (varios ids pueden repetirse)
        dembeddings = np.zeros_like(self.embeddings)
        np.add.at(dembeddings, ids, demb)

        return {
            "embeddings": dembeddings,
            "W_q": dW_q, "W_k": dW_k, "W_v": dW_v,
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "W_out": dW_out, "b_out": db_out,
        }

    def aplicar_gradientes(self, grads, lr):
        for nombre, grad in grads.items():
            valor_actual = getattr(self, nombre)
            grad = np.clip(grad, -5.0, 5.0)  # recorte de gradiente, estabiliza el entrenamiento
            setattr(self, nombre, valor_actual - lr * grad)

    def aplicar_gradientes_adam(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """Optimizador Adam (Kingma & Ba, 2015), implementado a mano: momento
        de primer orden (media móvil del gradiente) y segundo orden (media
        móvil del gradiente al cuadrado), con corrección de sesgo. Converge
        notablemente más rápido que el SGD plano de `aplicar_gradientes`
        para este tipo de pérdidas ruidosas de secuencias cortas."""
        self._adam_t += 1
        t = self._adam_t
        for nombre, grad in grads.items():
            grad = np.clip(grad, -5.0, 5.0)
            valor_actual = getattr(self, nombre)

            if nombre not in self._adam_m:
                self._adam_m[nombre] = np.zeros_like(valor_actual)
                self._adam_v[nombre] = np.zeros_like(valor_actual)

            self._adam_m[nombre] = beta1 * self._adam_m[nombre] + (1 - beta1) * grad
            self._adam_v[nombre] = beta2 * self._adam_v[nombre] + (1 - beta2) * (grad ** 2)

            m_hat = self._adam_m[nombre] / (1 - beta1 ** t)
            v_hat = self._adam_v[nombre] / (1 - beta2 ** t)

            setattr(self, nombre, valor_actual - lr * m_hat / (np.sqrt(v_hat) + eps))

    # -- entrenamiento de un paso (forward + loss + backward + update) -------------

    def paso_entrenamiento(self, ids_entrada, ids_objetivo, lr, optimizador="adam"):
        cache = {}
        logits = self.forward_rapido(ids_entrada, cache=cache)
        loss, _ = self.perdida_entropia_cruzada(logits, ids_objetivo)
        grads = self.backward(cache, ids_objetivo)
        if optimizador == "adam":
            self.aplicar_gradientes_adam(grads, lr)
        else:
            self.aplicar_gradientes(grads, lr)
        return loss

    # -- generación --------------------------------------------------------------

    def generar(self, prompt, max_nuevos=6, temperatura=1.0, max_contexto=16, top_k=None):
        """Genera texto de forma autoregresiva. `top_k`, si se indica,
        restringe el muestreo a los `top_k` tokens más probables en cada
        paso (renormalizando sus probabilidades) en vez de muestrear sobre
        todo el vocabulario — reduce la probabilidad de elegir tokens muy
        improbables y suele dar salidas algo más coherentes."""
        ids = self.tokenizer.codificar(prompt)
        if not ids:
            ids = [self.tokenizer.stoi.get("hola", self.tokenizer.stoi["<UNK>"])]

        for _ in range(max_nuevos):
            contexto_ids = ids[-max_contexto:]
            logits = self.forward_rapido(contexto_ids)
            ultimo = logits[-1] / max(temperatura, 1e-6)
            probs = self.softmax_vectorizado(ultimo)

            if top_k is not None and 0 < top_k < len(probs):
                indices_top = np.argpartition(probs, -top_k)[-top_k:]
                probs_top = probs[indices_top]
                probs_top = probs_top / probs_top.sum()
                siguiente = int(np.random.choice(indices_top, p=probs_top))
            else:
                siguiente = int(np.random.choice(len(probs), p=probs))

            ids.append(siguiente)

        return self.tokenizer.decodificar(ids)

    # -- persistencia completa del modelo -----------------------------------------

    def state_dict(self):
        return {
            "embeddings": self.embeddings.tolist(),
            "W_q": self.W_q.tolist(), "W_k": self.W_k.tolist(), "W_v": self.W_v.tolist(),
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W_out": self.W_out.tolist(), "b_out": self.b_out.tolist(),
            "adam": {
                "t": self._adam_t,
                "m": {k: v.tolist() for k, v in self._adam_m.items()},
                "v": {k: v.tolist() for k, v in self._adam_v.items()},
            },
        }

    def load_state_dict(self, datos):
        for nombre, valor in datos.items():
            if nombre == "adam":
                continue
            setattr(self, nombre, np.array(valor))

        adam = datos.get("adam")
        if adam:
            self._adam_t = adam.get("t", 0)
            self._adam_m = {k: np.array(v) for k, v in adam.get("m", {}).items()}
            self._adam_v = {k: np.array(v) for k, v in adam.get("v", {}).items()}
