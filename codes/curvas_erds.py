
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
sesion=1
sfreq = 512

channels_to_drop = ["FP1","FP2","FPz","Fz","F8","F7","AF3","AF4","AF5","AF7","AF8","T7","T8","F9","F10"]
pick = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6",]

eeg_concatenados = concatenateEEGs(sujeto, sesion, apply_ica=False).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")
# eeg_concatenados.plot_sensors(kind="topomap",show_names=True) ##probar con kind="3d"
# plotEEG(eeg_concatenados, show=True, scalings=40, bad_color = "red")

l_freq, h_freq = 8, 13
eeg_concatenados.filter(l_freq=l_freq, h_freq=h_freq,
           picks='eeg', 
           method='fir', 
           phase='zero-double', 
           fir_window='hamming',
           filter_length='auto')

##graficamos el espectro de potencia de los datos
# eeg_concatenados.compute_psd(fmax=100).plot(picks="data", exclude="bads", amplitude=True)

## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = -3, 4
event_ids = dict(IZQUIERDA=1, DERECHA=2)
epocas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True)

raw_eventos = mne.events_from_annotations(eeg_concatenados, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

# epocas.plot(scalings = 80,show=True, block=True,
#                           events=eventos,
#                           event_id=event_ids,
#                           event_color=dict(IZQUIERDA="red", DERECHA="blue"))

##separo los datos de cada clase
c3_index, c4_index = eeg_concatenados.ch_names.index("C3"), eeg_concatenados.ch_names.index("C4")
clase_izquierda = epocas["IZQUIERDA"]
clase_derecha = epocas["DERECHA"]
clase_izquierda_avg = clase_izquierda.average()
clase_derecha_avg= clase_derecha.average()


## 3. ************************ CURVAS ERDS% USANDO HILBERT ************************
# Graficar curva ERDS para un canal específico (por ejemplo, C3)
baseline = Baseline((-1.5, -0.5))  # Intervalo de tiempo para el baseline
erds_izq = getHilbertERDS(clase_izquierda, baseline, apply_smooth=True, window_smoothing=51, mean_trials=True)
erds_der = getHilbertERDS(clase_derecha, baseline, apply_smooth=True, window_smoothing=51, mean_trials=True)

##reemlazo data en clase_izquierda_avg y clase_derecha_avg
clase_izquierda_avg.data = erds_izq
clase_derecha_avg.data = erds_der
# erds_izq_avg= np.mean(erds_izq, axis=0)
# erds_der_avg= np.mean(erds_der, axis=0)

ti, tf = -1, 2.5
times = clase_izquierda.times
idx = np.where((times >= ti) & (times <= tf))[0]

plt.figure(figsize=(10, 5))
plt.plot(times[idx], erds_izq[c3_index, idx], label=f'ERDS% IZQUIERDA')
plt.plot(times[idx], erds_der[c3_index, idx], label=f'ERDS% DERECHA')
plt.axvline(0, color='k', linestyle='--', label='Cue onset')
plt.axhline(0, color='grey', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('ERDS (%)')
plt.title(f'Curva EDRS% C3')
plt.legend()
# plt.xlim(-1, 3)
plt.ylim(-50, 50)
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(times[idx], erds_izq[c4_index, idx], label=f'ERDS% IZQUIERDA')
plt.plot(times[idx], erds_der[c4_index, idx], label=f'ERDS% DERECHA')
plt.axvline(0, color='k', linestyle='--', label='Cue onset')
plt.axhline(0, color='grey', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('ERDS (%)')
plt.title(f'Curva EDRS% C4')
plt.legend()
# plt.xlim(-1, 3)
plt.ylim(-50, 50)
plt.grid()
plt.show()

clase_izquierda_avg.plot_topomap(times=[-0.5,0,0.5,1,1.5,2], ch_type="eeg", show=True, colorbar=True)
clase_derecha_avg.plot_topomap(times=[-0.5,0,0.5,1,1.5,2], ch_type="eeg", show=True, colorbar=True)