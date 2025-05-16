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

tipo_sesion = "Ejecutada" if sesion == 1 else "Imaginada"

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
# eeg_concatenados.compute_psd(fmax=100).plot(picks="data", exclude="bads", amplitude=True)
# eeg_concatenados.plot_psd(fmin=0, fmax=100, picks='eeg', average=True, show=True)
# eeg_concatenados.plot_psd_topo(fmin=6, fmax=40,fig_facecolor="white", color="red", axis_facecolor="white")

## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = -3, 4
event_ids = dict(IZQUIERDA=1, DERECHA=2)
epocas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True,reject={"eeg":80})

raw_eventos = mne.events_from_annotations(eeg_concatenados, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])
epocas.plot(scalings = 40,show=True, block=True,
                          events=eventos,
                          event_id=event_ids,
                          event_color=dict(IZQUIERDA="red", DERECHA="blue"))

eventos=epocas.events
labels=eventos[:,2]

csp = CSP(n_components=4, reg=None, log=None, norm_trace=False, transform_into="csp_space")
X = epocas.get_data(copy=False)#.astype(np.float64)
y=labels

csp.fit(X,y)
X_transformed = csp.transform(X)
csp.plot_patterns(epocas.info, ch_type='eeg', units='Patterns (a.u.)', size=1.5)
csp.plot_filters(epocas.info, ch_type='eeg', units='Patterns (a.u.)', size=1.5)

# info = epocas.info.copy()  # Información original de los datos
raw_info = epocas.info.copy()
epocas.tmin

info = mne.create_info(ch_names=[f"CSP{i+1}" for i in range(X_transformed.shape[1])],  # Nombres para los nuevos "canales"
                       sfreq=epocas.info['sfreq'],  # La frecuencia de muestreo es la misma que en el objeto original
                       ch_types='eeg')  # Tipo de canal es EEG

event_ids = dict(IZQUIERDA=1, DERECHA=2)
epochs_csp = mne.EpochsArray(data=X_transformed, info=info,
                             events=eventos, event_id=event_ids,
                             tmin=epocas.tmin,reject={"eeg":10},)

epochs_csp[18:24].plot(scalings=5,events=eventos,event_id=event_ids,)

##PSD
# epochs_csp.plot_psd(fmin=0, fmax=100, picks='eeg', show=True)

##separo los datos de cada clase
componentes = ['CSP1', 'CSP2', 'CSP3', 'CSP4']
# c3_index = epochs_csp.ch_names.index(componentes[0])
# c4_index = eeg_concatenados.ch_names.index(electrodos[1])
epochs_csp_izq = epochs_csp["IZQUIERDA"]
epochs_csp_der = epochs_csp["DERECHA"]
clase_izquierda_avg = epochs_csp_izq.average()
clase_derecha_avg= epochs_csp_der.average()


## 3. ************************ CURVAS ERDS% USANDO HILBERT ************************
# Graficar curva ERDS para un canal específico (por ejemplo, C3)
baseline = Baseline((-3, 0))  # Intervalo de tiempo para el baseline
erds_izq = getHilbertERDS(epochs_csp_izq, baseline, apply_smooth=True, window_smoothing=51, mean_trials=True)
erds_der = getHilbertERDS(epochs_csp_der, baseline, apply_smooth=True, window_smoothing=51, mean_trials=True)



ti, tf = -0.5, 2.5
times = epochs_csp_izq.times
comp = 2
idx = np.where((times >= ti) & (times <= tf))[0]
plt.figure(figsize=(10, 5))
plt.plot(times[idx], erds_izq[comp, idx], label="IZQUIERDA")
plt.plot(times[idx], erds_der[comp, idx], label="DERECHA")
plt.axvline(0, color="k", linestyle="--", label="Cue onset")
plt.axhline(0, color="grey", linestyle="--")
plt.xlabel("Tiempo (s)")
plt.ylabel("ERDS (%)")
plt.title(f"EDRS% {componentes[comp]} - Banda ({l_freq}-{h_freq}) Hz - Sesión {tipo_sesion}", fontsize=14)
plt.legend(loc="upper right")
# plt.xlim(-1, 3)
plt.ylim(-70, 70)
plt.grid()
plt.show()


data = epochs_csp[9:11].get_data().swapaxes(1, 0).reshape(4, 2*3687)

#data tiene shape 4,7374
##genero un gráfico de 4 filas y 1 columna donde se grafican los 4 componentes
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8), sharex=True)
for i in range(4):
    axes[i].plot(data[i], label=f"{componentes[i]}")
    axes[i].axvline(0, color="k", linestyle="--", label="Cue onset")
    axes[i].axhline(0, color="grey", linestyle="--")
    axes[i].set_ylabel("ERDS (%)")
    axes[i].legend(loc="upper right")
    # plt.xlim(-1, 3)
    # axes[i].set_ylim(-70, 70)
    axes[i].grid()
axes[3].set_xlabel("Tiempo (s)")
plt.suptitle(f"EDRS% {componentes} - Banda ({l_freq}-{h_freq}) Hz - Sesión {tipo_sesion}", fontsize=14)
plt.tight_layout()
plt.show()