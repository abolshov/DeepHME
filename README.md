# DeepHME
Heavy Mass Estimator ([HME](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.035007)), but based on deep neural network. 
Package can be installed via
```
pip install DeepHME
```
Example of usage is located in ([`example.py`](https://github.com/cms-flaf/DeepHME/blob/main/example.py)), data file is provided in [`data`](https://github.com/cms-flaf/DeepHME/tree/main/data). To instantiate `DeepHME` object one must provide the following named arguments to the constructor:
1. `model_name`: string with the name of the model to be used for inference. Currently available models are located in `models/` directory together with their training configuration files. In case of selecting model absent in `models/` an exception is thrown.
2. `channel`: string with the channel name. Must be uppercase. Allowed options are `SL` and `DL`. In the case of illegal value an exception will be thrown.
3. `return_errors`: boolean flag indicating whether or not to return per-event errors computed by the model. Note that at construction stage it is checked that selected model is capable of computed errors. If it is not, an exception will be thrown.
It is recomnended to use `predict_quantiles3D_DL_v8` for double lepton channel and `predict_quantiles3D_SL_v3` for single lepton channel. Additionally, for bettter performance it is recommended to ensure that event contains at least 2 AK4 jets (in what follows referred to as jets) with `pt > 20.0` and `eta < 2.5` or at least one AK8 jet (in what follows referred to as fatjet) with `pt > 200.0` and `eta < 2.5` and two leptons with `pt > 5.0` for double lepton channel and at least 2 AK4 jets with `pt > 20.0` and `eta < 2.5` and at least 2 AK4 jets with `pt > 20.0` and `eta < 5.0` or least one AK8 jet with `pt > 200.0` and `eta < 2.5` and least one AK8 jet with `pt > 200.0` and `eta < 5.0` and one lepton with `pt > 5.0` for single lepton channel. AK4 jet candidates must satisfy $\Delta R \ge 0.4$ between AK4 jet and leptons. AK8 jet candidates must satisfy $\Delta R \ge 0.8$ between AK4 jet and leptons. Electrons must satisfy `abs(Electron_dz) < 0.1 && abs(Electron_dxy) < 0.05 && Electron_sip3d <= 8 && Electron_miniPFRelIso_all < 0.4 && Electron_mvaIso_WP90 &&  Electron_mvaIso_WP80`. Muons must satisfy `abs(Muon_dz) < 0.1 && abs(Muon_dxy) < 0.05 && abs(Muon_dxy) < 0.05 && Muon_sip3d <= 8 && Muon_pfIsoId >= 1 && Muon_looseId && Muon_tightId`. Object variable naming is consistent with [NanoAODv12](https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv12/2022/2023/doc_DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8_Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2.html) for Run 3 2022/2023 eras.
Example of initialziation is 
```
estimator = DeepHME(model_name='predict_quantiles3D_DL_v8', channel=ch, return_errors=True)
```
For inference use method `predict`. It takes following arguments:
1. `event_id`: akward array of event ids
2. `lep1_pt`: akward array of lepton 1 pt 
3. `lep1_eta`: akward array of lepton 1 eta 
4. `lep1_phi`: akward array of lepton 1 phi
5. `lep1_mass`: akward array of lepton 1 phi mass
6. `lep2_pt:` akward array of lepton 2 pt
7. `lep2_eta`: akward array of lepton 2 eta
8. `lep2_phi`: akward array of lepton 2 phi
9. `lep2_mass`: akward array of lepton 2 mass
10. `met_pt`: akward array of met pt 
11. `met_phi`: akward array of met phi 
12. `jet_pt`: akward array of jet pt 
13. `jet_eta`: akward array of jet eta 
14. `jet_phi`: akward array of jet phi 
15. `jet_mass`: akward array of jet mass
16. `jet_btagPNetB`: akward array of jet btagPNetB scores
17. `jet_btagPNetCvB`: akward array of jet btagPNetCvB scores
18. `jet_btagPNetCvL`: akward array of jet btagPNetCvL scores 
19. `jet_btagPNetCvNotB`: akward array of jet btagPNetCvNotB scores 
20. `jet_btagPNetQvG`: akward array of jet btagPNetQvG scores
21. `jet_PNetRegPtRawCorr`: akward array of jet PNetRegPtRawCorr 
22. `jet_PNetRegPtRawCorrNeutrino`: akward array of jet PNetRegPtRawCorrNeutrino 
23. `jet_PNetRegPtRawRes`: akward array of jet PNetRegPtRawRes
24. `fatjet_pt`: akward array of fatjet pt
25. `fatjet_eta`: akward array of fatjet eta
26. `fatjet_phi`: akward array of fatjet phi
27. `fatjet_mass`: akward array of fatjet mass 
28. `fatjet_particleNet_QCD`: akward array of fatjet particleNet_QCD score
29. `fatjet_particleNet_XbbVsQCD`: akward array of fatjet particleNet_XbbVsQCD score
30. `fatjet_particleNetWithMass_QCD`: akward array of fatjet particleNetWithMass_QCD score 
31. `fatjet_particleNetWithMass_HbbvsQCD`: akward array of fatjet particleNetWithMass_HbbvsQCD score
32. `fatjet_particleNet_massCorr`: akward array of fatjet particleNet_massCorr
33. `output_format`: string with desired output format. Currently two output options are supported: `mass` and `p4`. If set to `mass`, will return a numpy array of masses. If set to `p4`, will return numpy array of shape `(n_events, 8)`. First 4 entries of `axis=1` are `px`, `py`, `pz` and `E` of H->VV, next for - `px`, `py`, `pz` and `E` of H->bb in this order. Defaults to `mass`.
All arguments except for `output_format` default to `None`. If any of them is not provided, an excpetion will be thrown. It is only allowed to leave `lep2*` variables as `None` if `channel` set to `SL`. Usage example is 
```
mass, errors = estimator.predict(event_id=arr1,
                                lep1_pt=arr2, 
                                lep1_eta=arr3, 
                                lep1_phi=arr4, 
                                lep1_mass=arr5,
                                lep2_pt=arr6, 
                                lep2_eta=arr7, 
                                lep2_phi=arr8, 
                                lep2_mass=arr9,
                                met_pt=arr10, 
                                met_phi=arr11,
                                jet_pt=arr12, 
                                jet_eta=arr13, 
                                jet_phi=arr14, 
                                jet_mass=arr15, 
                                jet_btagPNetB=arr16, 
                                jet_btagPNetCvB=arr17, 
                                jet_btagPNetCvL=arr18, 
                                jet_btagPNetCvNotB=arr19, 
                                jet_btagPNetQvG=arr20,
                                jet_PNetRegPtRawCorr=arr21, 
                                jet_PNetRegPtRawCorrNeutrino=arr22, 
                                jet_PNetRegPtRawRes=arr23,
                                fatjet_pt=arr24, 
                                fatjet_eta=arr25, 
                                fatjet_phi=arr26, 
                                fatjet_mass=arr27,
                                fatjet_particleNet_QCD=arr28, 
                                fatjet_particleNet_XbbVsQCD=arr29, 
                                fatjet_particleNetWithMass_QCD=arr30, 
                                fatjet_particleNetWithMass_HbbvsQCD=arr31, 
                                fatjet_particleNet_massCorr=arr32,
                                output_format='mass')
```
For currently available models, shape of returned errors is `(n_events, 1)` if `output_format` is set to `'mass'` and `(n_events, 6)` if it is set to `'p4'`. 