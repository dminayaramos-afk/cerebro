# Mi Cerebro IA

Un modelo de lenguaje (transformer) **construido enteramente desde cero**,
sin PyTorch, TensorFlow ni ningun framework de autodiferenciacion — solo
NumPy puro. Incluye tokenizador propio, retropropagacion manual derivada a
mano, un optimizador Adam implementado desde cero, un sistema de memoria
persistente, y un agente con herramientas controladas y limites de
seguridad explicitos.

## ⚠️ Alcance honesto de este proyecto

Esto **no** es un LLM competitivo ni un chatbot con conocimiento general.
Es un proyecto educativo para entender e implementar de primera mano como
funciona un transformer por dentro: atencion, backpropagation a traves de
softmax/atencion/residuales, y un ciclo de entrenamiento real. El modelo
por defecto tiene ~10.000 parametros (configurable) y se entrena sobre un
corpus de unas pocas frases — pensado para correr en CPU con RAM limitada
(desarrollado y probado en un Asus X550LC), no para generar texto de
calidad general.

## Que hay implementado de verdad (no simulado)

- **Tokenizador** (`core/tokenizer.py`): a nivel de palabra, vocabulario
  fijo con tokens especiales (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`),
  persistible a JSON.
- **Transformer** (`core/transformer/optimized_llm.py`): embeddings
  entrenables, auto-atencion de una cabeza con mascara causal, bloque
  feed-forward con ReLU y conexiones residuales.
- **Backpropagation manual real**: los gradientes de `backward()` son los
  gradientes exactos de la perdida de entropia cruzada respecto a cada
  matriz de pesos, derivados a mano con la regla de la cadena a traves de
  softmax, atencion, FFN y residuales — no hay actualizaciones inventadas.
- **Optimizador Adam** (ademas de SGD plano), implementado desde cero con
  momentos de primer/segundo orden y correccion de sesgo.
- **Entrenamiento con validacion real** (`train.py`): division
  train/val, perdida de validacion calculada sin actualizar pesos, y
  guardado del mejor checkpoint segun esa validacion (actua como
  *early stopping* automatico).
- **Evolucion de hiperparametros** (`evolve.py`): entrena varios modelos
  candidato con distintas combinaciones de learning rate/semilla, mide su
  perdida de validacion real, y solo sustituye el checkpoint principal si
  el candidato ganador generaliza mejor que lo que ya habia.
- **Memoria persistente** (`memory/advanced_memory.py`): contexto de
  corto plazo en memoria + memoria de largo plazo persistida a JSON con
  metadatos (categoria, importancia, fecha).
- **Agente controlado** (`agent/`): ejecuta herramientas con limites
  explicitos de pasos y de numero de llamadas, y verificacion de
  permisos por herramienta. La calculadora se evalua con `ast` (nunca
  `eval()`), y el acceso a archivos esta confinado a un `WORKSPACE_DIR`
  con proteccion contra *path traversal*.
- **Benchmark real** (`benchmarks/benchmark_engine.py`): cuenta de
  parametros exacta, uso de RAM/CPU, tokens por segundo — sin
  simulaciones ni llamadas a APIs externas.

## Instalacion

```bash
git clone https://github.com/dminayaramos-afk/cerebro.git
cd cerebro
pip install -r requirements.txt
```

Unica dependencia real: `numpy`. `psutil` es opcional (si esta instalado,
el benchmark reporta RAM/CPU reales; si no, esos campos quedan en 0).

## Uso

### Hablar con el modelo (chat local)

```bash
python3 main.py
```

Comandos disponibles dentro del chat: `/help`, `/status`, `/memory`, `/exit`.
Si existe un checkpoint entrenado (`checkpoints/checkpoint.json`), se
carga automaticamente en vez de usar pesos aleatorios.

### Entrenar

```bash
python3 train.py --epochs 20
python3 train.py --epochs 20 --resume          # continuar desde el checkpoint
python3 train.py --epochs 20 --corpus data/corpus.txt   # corpus alternativo
```

### Evolucionar (busqueda de hiperparametros)

```bash
python3 evolve.py --candidatos 4 --epocas-candidato 5 --epocas-final 20
```

### Evaluar / benchmark

```bash
python3 evaluate.py
```

### Ejecutar las pruebas de integridad

```bash
python3 tests/tests_final.py
```

Cubren: forward/backward/generacion, que el backward realmente reduce la
perdida tras varios pasos, persistencia exacta del modelo (roundtrip),
memoria, agente y herramientas, y dos pruebas de seguridad explicitas: que
la calculadora rechaza codigo Python arbitrario y que el acceso a archivos
bloquea intentos de salir del workspace.

## Configuracion (`config.py`)

| Variable | Que controla |
|---|---|
| `MAX_VOCAB_SIZE`, `DIM_EMBEDDING`, `DIM_QK`, `DIM_FFN` | Tamano del modelo |
| `MAX_SEQ_LEN` | Longitud maxima de secuencia por ventana de entrenamiento |
| `TASA_APRENDIZAJE`, `OPTIMIZADOR` | Hiperparametros de entrenamiento (`"adam"` o `"sgd"`) |
| `MAX_AGENT_STEPS`, `MAX_TOOL_CALLS` | Limites de seguridad del agente |
| `WORKSPACE_DIR` | Directorio al que el agente puede leer/escribir archivos |

## Estructura del proyecto

```
config.py                          # configuracion central
main.py                            # chat interactivo
train.py                           # entrenamiento con validacion
evolve.py                          # busqueda de hiperparametros
evaluate.py                        # benchmark/diagnostico
trainer_engine.py                  # motor de entrenamiento (usa el modelo)
core/tokenizer.py                  # tokenizador desde cero
core/transformer/optimized_llm.py  # transformer + backward + Adam
memory/advanced_memory.py          # memoria corto/largo plazo persistente
agent/controlled_agent.py          # agente con limites de seguridad
agent/tools/registry.py            # herramientas (calculadora, reloj, archivos)
benchmarks/benchmark_engine.py     # benchmark real (params, RAM, CPU, tok/s)
tests/tests_final.py               # pruebas de integridad end-to-end
data/corpus.txt                    # corpus de ejemplo para entrenar
```

## Notas sobre por que el modelo "no parece muy inteligente"

Con un vocabulario de 250 tokens, dimension de embedding 16 y un corpus de
10 frases, el modelo no tiene capacidad ni datos para generar texto
coherente de forma consistente — y es esperable ver mucho sobreajuste si
se entrena demasiadas epocas sobre un corpus tan pequeno (por eso
`train.py` y `evolve.py` solo se quedan con el checkpoint de mejor
perdida de **validacion**, no de entrenamiento). El valor del proyecto no
esta en la calidad del texto generado, sino en que cada componente
(tokenizador, atencion, backward, Adam, entrenamiento, evolucion) es una
implementacion real y verificable, no una simulacion.
