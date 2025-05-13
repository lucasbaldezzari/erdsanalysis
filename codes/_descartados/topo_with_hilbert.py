"""
La idea es poder replicar los mapas topográficos como hicieron en
"Optimizin Spatial Filter for Robust EEG Single-Trial Analysis" (2008)
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs, getHilbertERDS
from mne.time_frequency import tfr_morlet
from codes.utils import Baseline
from neuroiatools.DisplayData.plotEEG import plotEEG
from matplotlib.colors import TwoSlopeNorm
from codes.utils import LaplacianFilter, EnvolventeEEG, getEpochsBaseline

## 1. ******* Cargamos y concatenamos los datos para el sujeto y la sesión en cuestión *******
sujeto=4
sesion=2
sfreq = 512
baseline_rest = (-1.5, -0.5)
baseline = Baseline(baseline_rest)
montage = {'C3': ['C1', 'C5', 'CP3', 'FC3'],
           'C4': ['C2', 'C6', 'CP4', 'FC4']
           }
lap_filter = LaplacianFilter(montage)

channels_to_drop = ["FP1","FP2","FPz","Fz","F8","F7","AF3","AF4","AF5","AF7","AF8","T7","T8","F9","F10"]
pick = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6",]

eeg_concatenados = concatenateEEGs(sujeto, sesion, apply_ica=False).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")
# eeg_concatenados.plot_sensors(kind="topomap",show_names=True) ##probar con kind="3d"

lap_filter.apply(eeg_concatenados, inplace=True)
# plotEEG(eeg_concatenados, show=True, scalings=40, bad_color = "red")

l_freq, h_freq = 9, 13
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
epocas_concatenadas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True)

raw_eventos = mne.events_from_annotations(eeg_concatenados, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

# epocas_concatenadas.plot(scalings = 80,show=True, block=True,
#                           events=eventos,
#                           event_id=event_ids,
#                           event_color=dict(IZQUIERDA="red", DERECHA="blue"))

## 3. **************************** HILBERT ********************************
envelopeeeg = EnvolventeEEG(epocas_concatenadas, smoothing_window=512)
epocas_envelop = envelopeeeg.procesar(db=True)

c3_index, c4_index = eeg_concatenados.ch_names.index("C3"), eeg_concatenados.ch_names.index("C4")
epocas_envelop_izq = epocas_envelop["IZQUIERDA"].copy()
epocas_envelop_der = epocas_envelop["DERECHA"].copy()
envelop_izq_avg = epocas_envelop_izq.average()
envelop_der_avg = epocas_envelop_der.average()

izq_baseline = getEpochsBaseline(epocas_envelop_izq, baseline=baseline, times=epocas_envelop_izq.times)
der_baseline = getEpochsBaseline(epocas_envelop_der, baseline=baseline, times=epocas_envelop_der.times)

data_izq = izq_baseline.get_data(picks=["CP3", "CP4"]).mean(axis=0)
data_der = der_baseline.get_data(picks=["CP3", "CP4"]).mean(axis=0)

ti, tf = -1, 2.5
times = epocas_envelop_izq.times
idx = np.where((times >= ti) & (times <= tf))[0]
plt.plot(times[idx], data_izq[0,idx], label="Izquierda")
plt.plot(times[idx], data_der[0,idx], label="Derecha")
plt.legend()
plt.title("C3 Lap")
plt.show()

plt.plot(times[idx], data_izq[1,idx], label="Izquierda")
plt.plot(times[idx], data_der[1,idx], label="Derecha")
plt.legend()
plt.title("C4 Lap")
plt.show()