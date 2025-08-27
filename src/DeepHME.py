import numpy as np
import awkward as ak
import vector
import onnxruntime as ort
import yaml
import os

class DeepHME:
    def __init__(self, model_name):
        self._base_model_dir = 'models'
        self._model_dir = os.path.join(self._base_model_dir, model_name)
        self._train_cfg = {}
        with open(os.path.join(self._model_dir, 'params_model.yaml'), 'r') as train_cfg_file:
            self._train_cfg = yaml.safe_load(train_cfg_file)

    def predict(self):
        print(self._train_cfg)
    