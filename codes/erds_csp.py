# Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
# https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html#sphx-glr-auto-examples-decoding-decoding-csp-eeg-py

import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs, getHilbertERDS
from mne.time_frequency import tfr_morlet
from codes.utils import Baseline
from matplotlib.colors import TwoSlopeNorm
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

## 1. ******* Cargamos y concatenamos los datos para el sujeto y la sesión en cuestión *******
sujeto=8
sesion=2
sfreq = 512

channels_to_drop = ["FP1","FP2","FPz","Fz","F8","F7","AF3","AF4","AF5","AF7","AF8","T7","T8","F9","F10"]
pick = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6",]

eeg_concatenados = concatenateEEGs(sujeto, sesion, apply_ica=False).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")
# eeg_concatenados.plot_sensors(kind="topomap",show_names=True) ##probar con kind="3d"
# plotEEG(eeg_concatenados, show=True, scalings=40, bad_color = "red")

l_freq, h_freq = 7, 28
eeg_concatenados.filter(l_freq=7, h_freq=h_freq,
           picks='eeg', 
           method='fir', 
           phase='zero-double', 
           fir_window='hamming',
           filter_length='auto')

##graficamos el espectro de potencia de los datos
eeg_concatenados.compute_psd(fmax=100).plot(picks="data", exclude="bads", amplitude=True)
# eeg_concatenados.plot_psd(fmin=0, fmax=100, picks='eeg', average=True, show=True)
# eeg_concatenados.plot_psd_topo(fmin=6, fmax=40,fig_facecolor="white", color="red", axis_facecolor="white")

## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = -3, 4
event_ids = dict(IZQUIERDA=1, DERECHA=2)
epocas_concatenadas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True,)
eventos=epocas_concatenadas.events
labels=eventos[:,2]

csp = CSP(n_components=4, reg=None, log=None, norm_trace=False, transform_into="csp_space")
X = epocas_concatenadas.get_data(copy=False)#.astype(np.float64)
y=labels

csp.fit(X,y)
X_transformed = csp.transform(X)
csp.plot_patterns(epocas_concatenadas.info, ch_type='eeg', units='Patterns (a.u.)', size=1.5)
csp.plot_filters(epocas_concatenadas.info, ch_type='eeg', units='Patterns (a.u.)', size=1.5)

# info = epocas_concatenadas.info.copy()  # Información original de los datos
raw_info = epocas_concatenadas.info.copy()
epocas_concatenadas.tmin

info = mne.create_info(ch_names=[f"CSP{i+1}" for i in range(X_transformed.shape[1])],  # Nombres para los nuevos "canales"
                       sfreq=epocas_concatenadas.info['sfreq'],  # La frecuencia de muestreo es la misma que en el objeto original
                       ch_types='eeg')  # Tipo de canal es EEG

event_ids = dict(IZQUIERDA=1, DERECHA=2)
epochs_csp = mne.EpochsArray(data=X_transformed, info=info,
                             events=eventos, event_id=event_ids,
                             tmin=epocas_concatenadas.tmin)

epochs_csp.plot(scalings=20)

epochs_csp_izq = epochs_csp["IZQUIERDA"]
epochs_csp_der = epochs_csp["DERECHA"]

##PSD
epochs_csp_izq.plot_psd(fmin=0, fmax=100, picks='eeg', average=True, show=True)





# lda = LinearDiscriminantAnalysis()
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# pipeline = Pipeline([('CSP', csp), ('LDA', lda)])
# scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=-1)
# print(f"Accuracy promedio (10-fold): {np.mean(scores)*100:.2f}%")