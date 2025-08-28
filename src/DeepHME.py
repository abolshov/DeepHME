import numpy as np
import awkward as ak
import vector
import onnxruntime as ort
import yaml
import os
import pandas as pd

class DeepHME:
    def __init__(self, 
                 model_name='predict_quantiles3D_DL_v4',
                 output_format='p4',
                 channel='DL'):

        self._channel = channel
        self._output_format = output_format
        self._base_model_dir = 'models'
        self._model_dir = os.path.join(self._base_model_dir, model_name)
        self._train_cfg = {}
        with open(os.path.join(self._model_dir, 'params_model.yaml'), 'r') as train_cfg_file:
            self._train_cfg = yaml.safe_load(train_cfg_file)

        self._feature_map, self._object_count = self._gather_feature_info(self._train_cfg['input_names'])

        # FIXME: this must be two variables session for model for events with even event_id and separate session (and model) for odd
        self._session = ort.InferenceSession(os.path.join(self._model_dir, 'model.onnx'))
        self._model_input_name = self._session.get_inputs()[0].name
        self._model_output_names = [out.name for out in self._session.get_outputs()]

        # all these parameters must be split into two: event and odd models will have different configs
        quantiles = self._train_cfg['quantiles']
        self._is_quantile = quantiles is not None and len(quantiles) > 1 and 0.5 in quantiles
        self._standardize = self._train_cfg['standardize']
        self._input_means = self._train_cfg.get('input_train_means', None)
        self._input_scales = self._train_cfg.get('input_train_scales', None)
        self._target_means = self._train_cfg.get('target_train_means', None)
        self._target_scales = self._train_cfg.get('target_train_scales', None)

    def _compute_mass(self, central):
        hvv_en = central[:, 3]
        hbb_en = central[:, -1]

        hvv_p3 = central[:, :3]
        hbb_p3 = central[:, 4:-1]

        x_en = hvv_en + hbb_en
        x_p3 = hvv_p3 + hbb_p3
        x_mass_sqr = np.square(x_en) - np.sum(np.square(x_p3), axis=1)
        neg_mass = x_mass_sqr <= 0.0
        x_mass = np.sqrt(np.abs(x_mass_sqr))
        x_mass = np.where(neg_mass, -1.0, x_mass)
        return x_mass

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
        max_len = ak.max(ak.count(x, axis=1))
        x_padded = ak.fill_none(ak.pad_none(x, max_len), 0)
        return x_padded

    def _validate_arguments(self, args):
        for arg, val in args.items():
            if val is None:
                raise ValueError(f'Argument `{arg}` has illegal value `None`')

    def _concat_inputs(self, event_id, feature_values):
        df_dict = {'event_id': event_id}
        for object_name, cnt in self._object_count.items():
            feature_names = self._feature_map[object_name]
            if cnt > 1:
                df_dict.update({f'{object_name}_{i + 1}_{fn}': feature_values[object_name][fn][:, i] for fn in feature_names for i in range(cnt)})
            else:
                df_dict.update({f'{object_name}_{fn}': feature_values[object_name][fn] for fn in feature_names}) 
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

        jet_pt = self._add_padding(jet_pt)
        jet_eta = self._add_padding(jet_eta)
        jet_phi = self._add_padding(jet_phi)
        jet_mass = self._add_padding(jet_mass)
        jet_btagPNetB = self._add_padding(jet_btagPNetB)
        jet_btagPNetCvB = self._add_padding(jet_btagPNetCvB)
        jet_btagPNetCvL = self._add_padding(jet_btagPNetCvL)
        jet_btagPNetCvNotB = self._add_padding(jet_btagPNetCvNotB)
        jet_btagPNetQvG = self._add_padding(jet_btagPNetQvG)
        jet_PNetRegPtRawCorr = self._add_padding(jet_PNetRegPtRawCorr)
        jet_PNetRegPtRawCorrNeutrino = self._add_padding(jet_PNetRegPtRawCorrNeutrino)
        jet_PNetRegPtRawRes = self._add_padding(jet_PNetRegPtRawRes)
        fatjet_pt = self._add_padding(fatjet_pt)
        fatjet_eta = self._add_padding(fatjet_eta)
        fatjet_phi = self._add_padding(fatjet_phi)
        fatjet_mass = self._add_padding(fatjet_mass)
        fatjet_particleNet_QCD = self._add_padding(fatjet_particleNet_QCD)
        fatjet_particleNet_XbbVsQCD = self._add_padding(fatjet_particleNet_XbbVsQCD)
        fatjet_particleNetWithMass_QCD = self._add_padding(fatjet_particleNetWithMass_QCD)
        fatjet_particleNetWithMass_HbbvsQCD = self._add_padding(fatjet_particleNetWithMass_HbbvsQCD)
        fatjet_particleNet_massCorr = self._add_padding(fatjet_particleNet_massCorr)

        lep1_p4 = vector.zip({'pt': lep1_pt, 'eta': lep1_eta, 'phi': lep1_phi, 'mass': lep1_mass})
        lep2_p4 = vector.zip({'pt': lep2_pt, 'eta': lep2_eta, 'phi': lep2_phi, 'mass': lep2_mass})
        met_p4 = vector.zip({'pt': met_pt, 'eta': 0.0, 'phi': met_phi, 'mass': 0.0})
        jet_p4 = vector.zip({'pt': jet_pt, 'eta': jet_eta, 'phi': jet_phi, 'mass': jet_mass})
        fatjet_p4 = vector.zip({'pt': fatjet_pt, 'eta': fatjet_eta, 'phi': fatjet_phi, 'mass': fatjet_mass})

        num_jet = self._object_count['centralJet']
        num_fatjet = self._object_count['SelectedFatJet']

        jet_p4 = jet_p4[:, :num_jet]
        fatjet_p4 = fatjet_p4[:, :num_fatjet]

        jet_btagPNetB = jet_btagPNetB[:, :num_jet]
        jet_btagPNetCvB = jet_btagPNetCvB[:, :num_jet]
        jet_btagPNetCvL = jet_btagPNetCvL[:, :num_jet]
        jet_btagPNetCvNotB = jet_btagPNetCvNotB[:, :num_jet]
        jet_btagPNetQvG = jet_btagPNetQvG[:, :num_jet]
        jet_PNetRegPtRawCorr = jet_PNetRegPtRawCorr[:, :num_jet]
        jet_PNetRegPtRawCorrNeutrino = jet_PNetRegPtRawCorrNeutrino[:, :num_jet]
        jet_PNetRegPtRawRes = jet_PNetRegPtRawRes[:, :num_jet]        

        fatjet_particleNet_QCD = fatjet_particleNet_QCD[:, :num_fatjet]
        fatjet_particleNet_XbbVsQCD = fatjet_particleNet_XbbVsQCD[:, :num_fatjet]
        fatjet_particleNetWithMass_QCD = fatjet_particleNetWithMass_QCD[:, :num_fatjet]
        fatjet_particleNetWithMass_HbbvsQCD = fatjet_particleNetWithMass_HbbvsQCD[:, :num_fatjet]
        fatjet_particleNet_massCorr = fatjet_particleNet_massCorr[:, :num_fatjet]

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
                           'SelectedFatJet': fatjet_features,
                           'lep1': lep1_features,
                           'lep2': lep2_features,
                           'met': met_features}
        df = self._concat_inputs(event_id, object_features)

        df_even = df[df['event_id'] % 2 == 0]
        df_odd = df[df['event_id'] % 2 == 1]
        df_even = df_even.drop(['event_id'], axis=1)
        df_odd = df_odd.drop(['event_id'], axis=1)

        # reorder columns in dataframe to make sure features are in the same order as during training
        df_even = df_even[self._train_cfg['input_names']]
        df_odd = df_odd[self._train_cfg['input_names']]

        X_even = df_even.values
        X_odd = df_odd.values

        if self._standardize:
            X_odd -= self._input_means
            X_odd /= self._input_scales

        # outputs_even = self._session.run(self._model_output_names, {self._model_input_name: X_even.astype(np.float32)})
        outputs_odd = self._session.run(self._model_output_names, {self._model_input_name: X_odd.astype(np.float32)})
        
        central_odd = None
        if self._is_quantile:
            central_odd = np.array([out[:, 1] for out in outputs_odd[:-1]]).T
        else:
            central_odd = np.array(outputs_odd[:-1]).T

        if self._standardize:
            central_odd *= self._target_scales
            central_odd += self._target_means

        mass = None
        if self._output_format == 'mass':
            mass = self._compute_mass(central_odd)
            return mass
        return central
