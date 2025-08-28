import numpy as np
import awkward as ak
import vector
import onnxruntime as ort
import yaml
import os
import pandas as pd

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

        self._feature_map, self._object_count = self._gather_feature_info(self._train_cfg['input_names'])

    def _gather_feature_info(self, names):
        """
            returns two dicts: object -> list of feature names of that object, object -> number of objects of that type
        """
        object_features = {}
        object_count = {}
        unique_objects = list(set([name.split('_')[0] for name in names]))
        for uo in unique_objects:
            features = set(['_'.join(name.split('_')[2:]) if name.split('_')[1].isdigit() else '_'.join(name.split('_')[1:]) for name in names if uo in name])
            num_object_features = len([name for name in names if uo in name])
            num_obj = int(num_object_features/len(features))
            object_count[uo] = num_obj
            object_features[uo] = list(features)
        return object_features, object_count

    def _add_padding(self, x):
        max_len = ak.count(x, axis=1)
        x_padded = ak.fill_none(ak.pad_none(x, max_len), 0)
        return x_padded

    def _transform_kinematics(self, p4):
        pass

    def _validate_arguments(self, args):
        for arg, val in args.items():
            if val is None:
                raise ValueError(f'Argument `{arg}` has illegal value `None`')

    def _concat_inputs(self, event_id, p4s):
        pass

    def predict(self,
                event_id=None,
                lep1_pt=None, lep1_eta=None, lep1_phi=None, lep1_mass=None,
                lep2_pt=None, lep2_eta=None, lep2_phi=None, lep2_mass=None,
                met_pt=None, met_phi=None,
                jet_pt=None, jet_eta=None, jet_phi=None, jet_mass=None,
                fatjet_pt=None, fatjet_eta=None, fatjet_phi=None, fatjet_mass=None):

        self._validate_arguments(locals())

        for var in [jet_pt, jet_eta, jet_phi, jet_mass, fatjet_pt, fatjet_eta, fatjet_phi, fatjet_mass]:
            var = self._add_padding(var)

        lep1_p4 = vector.zip({'pt': lep1_pt, 'eta': lep1_eta, 'phi': lep1_phi, 'mass': lep1_mass})
        lep2_p4 = vector.zip({'pt': lep2_pt, 'eta': lep2_eta, 'phi': lep2_phi, 'mass': lep2_mass})
        met_p4 = vector.zip({'pt': met_pt, 'eta': 0.0, 'phi': met_phi, 'mass': 0.0})
        jet_p4 = vector.zip({'pt': jet_pt, 'eta': jet_eta, 'phi': jet_phi, 'mass': jet_mass})
        fatjet_p4 = vector.zip({'pt': fatjet_pt, 'eta': fatjet_eta, 'phi': fatjet_phi, 'mass': fatjet_mass})

        print(self._model_input_name)
        print(self._model_output_names)
    