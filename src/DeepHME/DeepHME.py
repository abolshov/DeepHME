import numpy as np
import awkward as ak
import vector
import onnxruntime as ort
import yaml
import os
import pandas as pd

class DeepHME:
    """
    Heavy Mass Estimator based on deep neural network for X->HH->bbWW mass estimation.
    Arguments:
        model_name: string with name of the model to be used for calculation. Available models are located in `models/` directory
                    Each subdirectory of `models/` contains folders with model names. These folders contain two model files in .onnx format
                    and .yaml files with parameters used for training of the even and odd models. Even model was trained on events with even ids,
                    odd - on events with odd ids. 
        channel: string specifyning channel. Options are `SL` and `DL`. Must be capital.
    """
    def __init__(self, 
                 model_name=None,
                 channel='DL'):

        if model_name is None:
            raise RuntimeError('Must provide name of the model to use. Available models can be found in `models` directory.')

        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pkg_dir = '/'.join(src_dir.split('/')[:-1])
        models_dir = os.path.join(pkg_dir, 'models')

        available_models = os.listdir(models_dir)
        if model_name not in available_models:
            raise RuntimeError(f'Model `{model_name}` is not available, currently available models are {available_models}.')

        if channel not in ['DL', 'SL']:
            raise RuntimeError(f'Channel `{channel}` is not supported, options are `SL`, `DL`.')

        self._channel = channel
        self._base_model_dir = models_dir
        self._model_dir = os.path.join(self._base_model_dir, model_name)
        
        self._train_cfg_odd = self._load_cfg(f'{model_name}_odd')
        self._train_cfg_even = self._load_cfg(f'{model_name}_even')

        feature_map_odd, object_count_odd = self._gather_feature_info(self._train_cfg_odd['input_names'])
        feature_map_even, object_count_even = self._gather_feature_info(self._train_cfg_even['input_names'])
        assert feature_map_even == feature_map_odd and object_count_odd == object_count_even, 'Config mismatch between even and odd models'
        self._feature_map, self._object_count = feature_map_even, object_count_even

        # even model trained on events with even ids => apply it to odds
        # odd model trained on events with odd ids => apply it to even
        self._session_even = ort.InferenceSession(os.path.join(self._model_dir, f'{model_name}_even.onnx'))
        self._session_odd = ort.InferenceSession(os.path.join(self._model_dir, f'{model_name}_odd.onnx'))
        input_name_odd = self._session_odd.get_inputs()[0].name
        input_name_even = self._session_even.get_inputs()[0].name
        assert input_name_even == input_name_odd, 'Input names mismatch between even and odd models'
        self._model_input_name = input_name_even

        output_names_even = [out.name for out in self._session_even.get_outputs()]
        output_names_odd = [out.name for out in self._session_odd.get_outputs()]
        assert output_names_even == output_names_odd, 'Output names mismatch between even and odd models'
        self._model_output_names = output_names_even

        quantiles_even = self._train_cfg_even['quantiles']
        quantiles_odd = self._train_cfg_odd['quantiles']
        assert quantiles_even == quantiles_odd, '`quantiles` mismatch between even and odd models'
        quantiles = quantiles_even
        self._is_quantile = quantiles is not None and len(quantiles) > 1 and 0.5 in quantiles
        
        standardize_even = self._train_cfg_even['standardize']
        standardize_odd = self._train_cfg_odd['standardize']
        assert standardize_even == standardize_odd, '`standardize` mismatch between even and odd models'
        self._standardize = standardize_even

        self._input_means_even = self._train_cfg_even.get('input_train_means', None)
        self._input_means_odd = self._train_cfg_odd.get('input_train_means', None)
        self._input_scales_even = self._train_cfg_even.get('input_train_scales', None)
        self._input_scales_odd = self._train_cfg_odd.get('input_train_scales', None)
        self._target_means_even = self._train_cfg_even.get('target_train_means', None)
        self._target_means_odd = self._train_cfg_odd.get('target_train_means', None)
        self._target_scales_even = self._train_cfg_even.get('target_train_scales', None)
        self._target_scales_odd = self._train_cfg_odd.get('target_train_scales', None)

    def _load_cfg(self, model_name):
        cfg = {}
        with open(os.path.join(self._model_dir, f'params_{model_name}.yaml'), 'r') as train_cfg_file:
            cfg = yaml.safe_load(train_cfg_file)
        return cfg

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
                fatjet_particleNet_QCD=None, fatjet_particleNet_XbbVsQCD=None, fatjet_particleNetWithMass_QCD=None, fatjet_particleNetWithMass_HbbvsQCD=None, fatjet_particleNet_massCorr=None,
                output_format='mass'):
        """
        Public interface for obtaining predictions of the model.
        Returns: np.array of mass of shape (num_events,) or np.array of p4 of shape (num_events, 8). 
                First 4 entries of axis=1 are `px`, `py`, `pz` and `E` of H->VV, next for - `px`, `py`, `pz` and `E` of H->bb in this order.
        Arguments:
            event_id: akward array of event ids
            lep1_pt: akward array of lepton 1 pt 
            lep1_eta: akward array of lepton 1 eta 
            lep1_phi: akward array of lepton 1 phi
            lep1_mass: akward array of lepton 1 phi mass
            lep2_pt: akward array of lepton 2 pt
            lep2_eta: akward array of lepton 2 eta
            lep2_phi: akward array of lepton 2 phi
            lep2_mass: akward array of lepton 2 mass
            met_pt: akward array of met pt 
            met_phi: akward array of met phi 
            jet_pt: akward array of jet pt 
            jet_eta: akward array of jet eta 
            jet_phi: akward array of jet phi 
            jet_mass: akward array of jet mass
            jet_btagPNetB: akward array of jet btagPNetB scores
            jet_btagPNetCvB: akward array of jet btagPNetCvB scores
            jet_btagPNetCvL: akward array of jet btagPNetCvL scores 
            jet_btagPNetCvNotB: akward array of jet btagPNetCvNotB scores 
            jet_btagPNetQvG: akward array of jet btagPNetQvG scores
            jet_PNetRegPtRawCorr: akward array of jet PNetRegPtRawCorr 
            jet_PNetRegPtRawCorrNeutrino: akward array of jet PNetRegPtRawCorrNeutrino 
            jet_PNetRegPtRawRes: akward array of jet PNetRegPtRawRes
            fatjet_pt: akward array of fatjet pt
            fatjet_eta: akward array of fatjet eta
            fatjet_phi: akward array of fatjet phi
            fatjet_mass: akward array of fatjet mass 
            fatjet_particleNet_QCD: akward array of fatjet particleNet_QCD score
            fatjet_particleNet_XbbVsQCD: akward array of fatjet particleNet_XbbVsQCD score
            fatjet_particleNetWithMass_QCD: akward array of fatjet particleNetWithMass_QCD score 
            fatjet_particleNetWithMass_HbbvsQCD: akward array of fatjet particleNetWithMass_HbbvsQCD score
            fatjet_particleNet_massCorr: akward array of fatjet particleNet_massCorr
            output_format: string with desired output format. Currently two output options are supported: `mass` and `p4`. If set to `mass`, 
                    will return a numpy array of masses. If set to `p4`, will return numpy array of shape
                    (n_events, 8). First 4 entries of axis=1 are `px`, `py`, `pz` and `E` of H->VV, next for - `px`, `py`, `pz` and `E` of H->bb in this order.
                    Defaults to `mass`.
        """

        args = locals()
        args.pop('self')
        args.pop('output_format')
        if self._channel == 'SL':
            args.pop('lep2_pt')
            args.pop('lep2_eta')
            args.pop('lep2_phi')
            args.pop('lep2_mass')
        self._validate_arguments(args)

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
        lep2_p4 = None
        if self._channel == 'DL':
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

        lep2_features = None
        if self._channel == 'DL':
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
        if self._channel == 'SL':
            object_features.pop('lep2')
        df = self._concat_inputs(event_id, object_features)

        # mask rows (events) with even event_id with True and odd with False
        # to return one array of masses or p4s in the same order as it was passed
        mask = np.where(df['event_id'] % 2 == 0, True, False)

        df_even = df[df['event_id'] % 2 == 0]
        df_odd = df[df['event_id'] % 2 == 1]
        df_even = df_even.drop(['event_id'], axis=1)
        df_odd = df_odd.drop(['event_id'], axis=1)

        # reorder columns in dataframe to make sure features are in the same order as during training
        df_even = df_even[self._train_cfg_even['input_names']]
        df_odd = df_odd[self._train_cfg_odd['input_names']]

        X_even = df_even.values
        X_odd = df_odd.values

        if self._standardize:
            X_odd -= self._input_means_odd
            X_odd /= self._input_scales_odd

            X_even -= self._input_means_odd
            X_even /= self._input_scales_even

        # even model trained on events with even ids => apply it to odds
        # odd model trained on events with odd ids => apply it to even
        outputs_even = self._session_odd.run(self._model_output_names, {self._model_input_name: X_even.astype(np.float32)})
        outputs_odd = self._session_even.run(self._model_output_names, {self._model_input_name: X_odd.astype(np.float32)})
        
        central_odd = None
        central_even = None
        if self._is_quantile:
            central_odd = np.array([out[:, 1] for out in outputs_odd[:-1]]).T
            central_even = np.array([out[:, 1] for out in outputs_even[:-1]]).T
        else:
            central_odd = np.array(outputs_odd[:-1]).T
            central_even = np.array(outputs_even[:-1]).T

        if self._standardize:
            central_odd *= self._target_scales_odd
            central_odd += self._target_means_odd

            central_even *= self._target_scales_even
            central_even += self._target_means_even

        match output_format:
            case 'mass':
                mass_odd = self._compute_mass(central_odd)
                mass_even = self._compute_mass(central_even)
                mass = np.full(len(df), -1)
                mass[mask] = mass_even
                mass[~mask] = mass_odd
                return mass
            case 'p4':
                assert central_even.shape[1] == central_odd.shape[1], 'P4 shape mismatch for even and odd events'
                num_p4_comp = central_even.shape[1]
                p4 = np.full((len(df), num_p4_comp), 1.0)
                p4[mask] = central_even
                p4[~mask] = central_odd
                return p4
            case _:
                raise RuntimeError(f'Illegal output format: `{output_format}`. Only `mass` or `p4` are supported.')
