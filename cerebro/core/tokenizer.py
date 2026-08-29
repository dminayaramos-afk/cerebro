# -*- coding: utf-8 -*-
"""
Tokenizador a nivel de palabra, construido enteramente desde cero (sin
librerías de NLP externas: solo `re` y `json` de la librería estándar).

Mantiene un vocabulario FIJO y FINITO (max_vocab_size) con tokens
especiales reservados. Cuando el vocabulario se llena, las palabras nuevas
se asignan al token <UNK> (fuera de vocabulario) en lugar de crecer sin
límite — importante en un equipo con RAM limitada, donde el tamaño del
vocabulario determina directamente el tamaño de la matriz de embeddings
(max_vocab_size × dim_embedding).
"""

import json
import os
import re

ESPECIALES = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

_PATRON_TOKEN = re.compile(r"[a-zA-ZáéíóúñÁÉÍÓÚÑüÜ0-9]+|[.,!?;:¿¡]")


class TokenizadorUnificado:
    """Tokenizador simple: separa palabras y signos de puntuación básicos,
    y les asigna un identificador entero estable dentro de un vocabulario
    de tamaño fijo."""

    def __init__(self, max_vocab_size=250, vocab_path=None):
        if max_vocab_size < len(ESPECIALES):
            raise ValueError(
                f"max_vocab_size debe ser al menos {len(ESPECIALES)} "
                f"(para los tokens especiales)."
            )
        self.max_vocab_size = max_vocab_size
        self.vocab_path = vocab_path
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

        for token in ESPECIALES:
            self._agregar_token(token)

        if vocab_path and os.path.exists(vocab_path):
            self.cargar_vocab(vocab_path)

    # -- construcción de vocabulario -------------------------------------

    def _tokenizar_texto(self, texto: str) -> list[str]:
        return _PATRON_TOKEN.findall(texto.lower())

    def _agregar_token(self, token: str) -> int:
        if token in self.stoi:
            return self.stoi[token]
        if len(self.stoi) >= self.max_vocab_size:
            return self.stoi["<UNK>"]
        idx = len(self.stoi)
        self.stoi[token] = idx
        self.itos[idx] = token
        return idx

    def registrar_textos(self, textos: list[str]) -> None:
        """Añade al vocabulario las palabras nuevas encontradas en `textos`,
        hasta llenar max_vocab_size. Llamar repetidamente (p.ej. con cada
        conversación nueva) permite que el vocabulario crezca con el uso,
        de forma similar a cómo el sistema de memoria acumula experiencia."""
        for texto in textos:
            for palabra in self._tokenizar_texto(texto):
                self._agregar_token(palabra)
        if self.vocab_path:
            self.guardar_vocab(self.vocab_path)

    # -- codificación / decodificación ------------------------------------

    def codificar(self, texto: str) -> list[int]:
        palabras = self._tokenizar_texto(texto)
        unk = self.stoi["<UNK>"]
        return [self.stoi.get(p, unk) for p in palabras]

    def decodificar(self, ids: list[int]) -> str:
        ocultos = {"<PAD>"}
        palabras = [
            self.itos.get(i, "<UNK>")
            for i in ids
            if self.itos.get(i, "<UNK>") not in ocultos
        ]
        return " ".join(palabras)

    # -- persistencia -------------------------------------------------------

    def guardar_vocab(self, path: str) -> None:
        directorio = os.path.dirname(path)
        if directorio:
            os.makedirs(directorio, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    def cargar_vocab(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
        # Los tokens especiales siempre deben mantener sus IDs 0..3.
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)
