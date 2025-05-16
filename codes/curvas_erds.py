import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs, getHilbertERDS
from mne.time_frequency import tfr_morlet
from codes.utils import Baseline
from matplotlib.colors import TwoSlopeNorm
from codes.utils import LaplacianFilter, EnvolventeEEG

## 1. ******* CARGAMOS Y CONCATENAMOS LOS DATOS PARA EL/LA SUJETO EN CUESTIÓN *******
sujeto=8
sesion=2
sfreq = 512

tipo_sesion = "Ejecutada" if sesion == 1 else "Imaginada"

channels_to_drop = ["FP1","FP2","FPz","Fz","F8","F7","AF3","AF4","AF5","AF7","AF8","T7","T8","F9","F10"]
pick = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6",]

eeg_concatenados = concatenateEEGs(sujeto, sesion, apply_ica=False).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")
# eeg_concatenados.plot_sensors(kind="topomap",show_names=True) ##probar con kind="3d"

# montage = {"C4": ["C2", "C6", "CP4", "FC4"],
#            "C3": ["C1", "C5", "CP3", "FC3"]
#            }

# lapfilter = LaplacianFilter(montage)
# lapfilter.apply(eeg_concatenados, inplace=True)

# plotEEG(eeg_concatenados, show=True, scalings=40, bad_color = "red")

l_freq, h_freq = 7, 32
eeg_concatenados.filter(l_freq=l_freq, h_freq=h_freq,
           picks="eeg", 
           method="fir", 
           phase="zero-double", 
           fir_window="hamming",
           filter_length="auto")

##graficamos el espectro de potencia de los datos
# eeg_concatenados.compute_psd(fmax=100).plot(picks="data", exclude="bads", amplitude=True)

## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = -3, 4
event_ids = dict(IZQUIERDA=1, DERECHA=2)
epocas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True,reject={"eeg":80})

raw_eventos = mne.events_from_annotations(eeg_concatenados, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

epocas.plot(scalings = 80,show=True, block=True,
                          events=eventos,
                          event_id=event_ids,
                          event_color=dict(IZQUIERDA="red", DERECHA="blue"))

electrodos = ["CP3","CP4"]
e1_index = eeg_concatenados.ch_names.index(electrodos[0])
e2_index = eeg_concatenados.ch_names.index(electrodos[1])
clase_izquierda = epocas["IZQUIERDA"]
clase_derecha = epocas["DERECHA"]
clase_izquierda_avg = clase_izquierda.average()
clase_derecha_avg= clase_derecha.average()

clase_izquierda_avg.plot(picks=electrodos, show=True, spatial_colors=True)
clase_izquierda_avg.plot_psd(fmax=100,picks=["C3","CP3","C4","CP4"], show=True, spatial_colors=True)

## 3. ************************ CURVAS ERDS% USANDO HILBERT ************************
# Graficar curva ERDS para un canal específico (por ejemplo, C3)
baseline = Baseline((-2, -1))  # Intervalo de tiempo para el baseline
base = baseline.apply(epocas.get_data().mean(axis=0),clase_izquierda.times)
erds_izq = getHilbertERDS(clase_izquierda, baseline, apply_smooth=True, window_smoothing=51, mean_trials=True)
erds_der = getHilbertERDS(clase_derecha, baseline, apply_smooth=True, window_smoothing=51, mean_trials=True)

ti, tf = -0.5, 2.5
times = clase_izquierda.times
idx = np.where((times >= ti) & (times <= tf))[0]
plt.figure(figsize=(10, 5))
plt.plot(times[idx], erds_izq[e1_index, idx], label="IZQUIERDA")
plt.plot(times[idx], erds_der[e1_index, idx], label="DERECHA")
plt.axvline(0, color="k", linestyle="--", label="Cue onset")
plt.axhline(0, color="grey", linestyle="--")
plt.xlabel("Tiempo (s)")
plt.ylabel("ERDS (%)")
plt.title(f"EDRS% {electrodos[0]} - Banda ({l_freq}-{h_freq}) Hz - Sesión {tipo_sesion}", fontsize=14)
plt.legend(loc="upper right")
# plt.xlim(-1, 3)
plt.ylim(-70, 70)
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(times[idx], erds_izq[e2_index, idx], label="IZQUIERDA")
plt.plot(times[idx], erds_der[e2_index, idx], label="DERECHA")
plt.axvline(0, color="k", linestyle="--", label="Cue onset")
plt.axhline(0, color="grey", linestyle="--")
plt.xlabel("Tiempo (s)")
plt.ylabel("ERDS (%)")
plt.title(f"EDRS% {electrodos[1]} - Banda ({l_freq}-{h_freq}) Hz - Sesión {tipo_sesion}")
plt.legend(loc="upper right")
# plt.xlim(-1, 3)
plt.ylim(-70, 70)
plt.grid()
plt.show()

chans = ["C2","C4","CP4"]#["C1","C3","CP3"]
data_izquierda=clase_izquierda_avg.get_data(picks=chans)
data_derecha=clase_derecha_avg.get_data(picks=chans)
# window = 512
# #aplicamos convolución a los datos
# for i, chan in enumerate(chans):
#     data_derecha[i] = np.convolve(data_derecha[i], np.ones(window)/window, mode='same')
#     data_izquierda[i] = np.convolve(data_izquierda[i], np.ones(window)/window, mode='same')

ti, tf = -1.5, 2.5
times = clase_izquierda.times
idx = np.where((times >= ti) & (times <= tf))[0]
##grafico de filas=nchans y 1 columna
fig, axes = plt.subplots(nrows=len(chans), ncols=1, figsize=(10, 5), sharex=True)
for i, chan in enumerate(chans):
    e_index = clase_derecha_avg.ch_names.index(chan)
    axes[i].plot(clase_derecha_avg.times[idx], data_izquierda[i, idx], label="IZQUIERDA")
    axes[i].plot(clase_derecha_avg.times[idx], data_derecha[i, idx], label="DERECHA")
    axes[i].axvline(0, color="k", linestyle="--", label="Cue onset")
    axes[i].axhline(0, color="grey", linestyle="--")
    axes[i].set_ylabel("ERDS (%)")
    axes[i].set_title(f"EDRS% {chan} - Banda ({l_freq}-{h_freq}) Hz - Sesión {tipo_sesion}", fontsize=14)
    axes[i].legend(loc="upper right")
    # plt.xlim(-1, 3)
    # axes[i].set_ylim(-70, 70)
    axes[i].grid()  
plt.xlabel("Tiempo (s)")
plt.tight_layout()
plt.show()

chans = ["C1","C3","CP3"]
fig, axes = plt.subplots(nrows=len(chans), ncols=1, figsize=(10, 5), sharex=True)
for i, chan in enumerate(chans):
    e_index = clase_derecha_avg.ch_names.index(chan)
    axes[i].plot(clase_derecha_avg.times[idx], data_izquierda[i, idx], label="IZQUIERDA")
    axes[i].plot(clase_derecha_avg.times[idx], data_derecha[i, idx], label="DERECHA")
    axes[i].axvline(0, color="k", linestyle="--", label="Cue onset")
    axes[i].axhline(0, color="grey", linestyle="--")
    axes[i].set_ylabel("ERDS (%)")
    axes[i].set_title(f"EDRS% {chan} - Banda ({l_freq}-{h_freq}) Hz - Sesión {tipo_sesion}", fontsize=14)
    axes[i].legend(loc="upper right")
    # plt.xlim(-1, 3)
    # axes[i].set_ylim(-70, 70)
    axes[i].grid()  
plt.xlabel("Tiempo (s)")
plt.tight_layout()
plt.show()