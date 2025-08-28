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

        # FIXME: this must be two variables session for model for events with even event_id and separate session (and model) for odd
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

    def _validate_arguments(self, args):
        for arg, val in args.items():
            if val is None:
                raise ValueError(f'Argument `{arg}` has illegal value `None`')

    def _concat_inputs(self, event_id, feature_values):
        df_dict = {}
        for object_name, cnt in self._object_count.items():
            features = self._feature_map[object_name]
            if cnt > 1:
                df_dict.update({f'{object_name}_{i + 1}_{f}': feature_values[f] for f in features for i in range(cnt)})
                for i in range(cnt):
            else:
                df_dict.update({f'{object_name}_{f}': feature_values[f] for f in features}) 
        return pd.DataFrame.from_dict(df_dict)

    def predict(self,
                event_id=None,
                lep1_pt=None, lep1_eta=None, lep1_phi=None, lep1_mass=None,
                lep2_pt=None, lep2_eta=None, lep2_phi=None, lep2_mass=None,
                met_pt=None, met_phi=None,
                jet_pt=None, jet_eta=None, jet_phi=None, jet_mass=None, 
                jet_btagPNetB=None, jet_btagPNetCvB=None, jet_btagPNetCvL=None, jet_btagPNetCvNotB=None, jet_btagPNetQvG=None,
                jet_PNetRegPtRawCorr=None, jet_PNetRegPtRawCorrNeutrino=None, jet_PNetRegPtRawRes=None,
                fatjet_pt=None, fatjet_eta=None, fatjet_phi=None, fatjet_mass=None,
                fatjet_particleNet_QCD=None, fatjet_particleNet_XbbVsQCD=None, fatjet_particleNetWithMass_QCD=None, fatjet_particleNetWithMass_HbbvsQCD=None, fatjet_particleNet_massCorr=None):

        self._validate_arguments(locals())

        vars_to_pad = [jet_pt, jet_eta, jet_phi, jet_mass, 
                       jet_btagPNetB, jet_btagPNetCvB, jet_btagPNetCvL, jet_btagPNetCvNotB, jet_btagPNetQvG,
                       jet_PNetRegPtRawCorr, jet_PNetRegPtRawCorrNeutrino, jet_PNetRegPtRawRes,
                       fatjet_pt, fatjet_eta, fatjet_phi, fatjet_mass,
                       fatjet_particleNet_QCD, fatjet_particleNet_XbbVsQCD, fatjet_particleNetWithMass_QCD, fatjet_particleNetWithMass_HbbvsQCD, fatjet_particleNet_massCorr]

        for var in vars_to_pad:
            var = self._add_padding(var)

        lep1_p4 = vector.zip({'pt': lep1_pt, 'eta': lep1_eta, 'phi': lep1_phi, 'mass': lep1_mass})
        lep2_p4 = vector.zip({'pt': lep2_pt, 'eta': lep2_eta, 'phi': lep2_phi, 'mass': lep2_mass})
        met_p4 = vector.zip({'pt': met_pt, 'eta': 0.0, 'phi': met_phi, 'mass': 0.0})
        jet_p4 = vector.zip({'pt': jet_pt, 'eta': jet_eta, 'phi': jet_phi, 'mass': jet_mass})
        fatjet_p4 = vector.zip({'pt': fatjet_pt, 'eta': fatjet_eta, 'phi': fatjet_phi, 'mass': fatjet_mass})

        jet_p4 = jet_p4[:, :self._object_count['centralJet']]
        fatjet_p4 = fatjet_p4[:, :self._object_count['SelectedFatJet']]

        jet_btagPNetB = jet_btagPNetB[: :self._object_count['centralJet']]
        jet_btagPNetCvB = jet_btagPNetCvB[: :self._object_count['centralJet']]
        jet_btagPNetCvL = jet_btagPNetCvL[: :self._object_count['centralJet']]
        jet_btagPNetCvNotB = jet_btagPNetCvNotB[: :self._object_count['centralJet']]
        jet_btagPNetQvG = jet_btagPNetQvG[: :self._object_count['centralJet']]
        jet_PNetRegPtRawCorr = jet_PNetRegPtRawCorr[: :self._object_count['centralJet']]
        jet_PNetRegPtRawCorrNeutrino = jet_PNetRegPtRawCorrNeutrino[: :self._object_count['centralJet']]
        jet_PNetRegPtRawRes = jet_PNetRegPtRawRes[: :self._object_count['centralJet']]        

        fatjet_particleNet_QCD = fatjet_particleNet_QCD[:, :self._object_count['SelectedFatJet']]
        fatjet_particleNet_XbbVsQCD = fatjet_particleNet_XbbVsQCD[:, :self._object_count['SelectedFatJet']]
        fatjet_particleNetWithMass_QCD = fatjet_particleNetWithMass_QCD[:, :self._object_count['SelectedFatJet']]
        fatjet_particleNetWithMass_HbbvsQCD = fatjet_particleNetWithMass_HbbvsQCD[:, :self._object_count['SelectedFatJet']]
        fatjet_particleNet_massCorr = fatjet_particleNet_massCorr[:, :self._object_count['SelectedFatJet']]

        jet_features = {'px': jet_p4.px,
                        'py': jet_p4.py,
                        'pz': jet_p4.pz, 
                        'E': jet_p4.E,
                        'btagPNetB': jet_btagPNetB,
                        'btagPNetCvB': jet_btagPNetCvB,
                        'btagPNetCvL': jet_btagPNetCvL,
                        'btagPNetCvNotB': jet_btagPNetCvNotB,
                        'btagPNetQvG': jet_btagPNetQvG,
                        'PNetRegPtRawCorr': jet_PNetRegPtRawCorr,
                        'PNetRegPtRawCorrNeutrino': jet_PNetRegPtRawCorrNeutrino,
                        'PNetRegPtRawRes': jet_PNetRegPtRawRes }
        
        fatjet_features = {'px': fatjet_p4.px,
                           'py': fatjet_p4.py,
                           'pz': fatjet_p4.pz, 
                           'E': fatjet_p4.E,
                           'particleNet_QCD': fatjet_particleNet_QCD,
                           'particleNet_XbbVsQCD': fatjet_particleNet_XbbVsQCD,
                           'particleNetWithMass_QCD': fatjet_particleNetWithMass_QCD,
                           'particleNetWithMass_HbbvsQCD': fatjet_particleNetWithMass_HbbvsQCD,
                           'particleNet_massCorr': fatjet_particleNet_massCorr }

        lep1_features = {'px': lep1_p4.px,
                         'py': lep1_p4.py,
                         'pz': lep1_p4.pz, 
                         'E': lep1_p4.E }

        lep2_features = {'px': lep2_p4.px,
                         'py': lep2_p4.py,
                         'pz': lep2_p4.pz, 
                         'E': lep2_p4.E }

        met_features = {'px': met_p4.px,
                        'py': met_p4.py }

        object_features = {'centralJet': jet_features,
                           'fatjet_features': fatjet_features,
                           'lep1': lep1_features,
                           'lep2': lep2_features,
                           'met': met_features}
        df = self._concat_inputs(event_id, object_features)

        df_even = df[df['event_id'] % 2 == 0]
        df_odd = df[df['event_id'] % 2 == 1]
        df_even.drop(['event_id'], axis=1, inplace=True)
        df_odd.drop(['event_id'], axis=1, inplace=True)

        # reorder columns in dataframe to make sure features are in the same order as during training
        df_even = df_even[self._train_cfg['input_names']]
        df_odd = df_odd[self._train_cfg['input_names']]

        X_even = df_even.values
        X_odd = df_odd.values

        outputs_even = self._session.run(self._model_output_names, {self._model_input_name: X_even.astype(np.float32)})
        outputs_odd = self._session.run(self._model_output_names, {self._model_input_name: X_odd.astype(np.float32)})


    