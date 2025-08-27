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

        self._session = ort.InferenceSession(os.path.join(self._model_dir, 'model.onnx'))
        self._model_input_name = self._session.get_inputs()[0].name
        self._model_output_names = [out.name for out in self._session.get_outputs()]

    def predict(self):
        print(self._model_input_name)
        print(self._model_output_names)
    