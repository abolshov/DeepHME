from .DeepHME import DeepHME

import os
PACKAGE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(PACKAGE_DIR, "models")

__all__ = ["DeepHME", "MODELS_DIR"]