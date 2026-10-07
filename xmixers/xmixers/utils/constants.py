import os

XMIXERS_DEBUG = os.getenv("XMIXERS_DEBUG", "False").lower() in {"1", "true", "yes"}

EMBED_DIM_BASE = 256
