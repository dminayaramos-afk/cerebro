import os

# Configuración adaptada para ASUS X550LC (CPU Local, RAM Limitada)
MAX_VOCAB_SIZE = 250
DIM_EMBEDDING = 16
DIM_QK = 16
DIM_FFN = 32
MAX_SEQ_LEN = 16
TASA_APRENDIZAJE = 0.01

# Agente y Límites de Seguridad
MAX_AGENT_STEPS = 4
MAX_TOOL_CALLS = 3
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/workspace"))

# Rutas de Persistencia
MEMORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "memory/memory_store.json"))
DIR_CHECKPOINTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints/"))
DIR_EVOLUTION = os.path.abspath(os.path.join(os.path.dirname(__file__), "learning/evolution/"))
