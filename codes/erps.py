# Código para analizar ERPs visuales usando MNE-Python
# https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html#sphx-glr-auto-tutorials-evoked-30-eeg-erp-py

import mne
import numpy as np
import matplotlib.pyplot as plt
from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
from codes.utils import concatenateEEGs

## 1. ******* Cargamos y concatenamos los datos para el sujeto y la sesión en cuestión *******
sujeto=1
sesion=1

channels_to_drop = ["FP1","FP2","FPz","Fz","F8","F7","AF4","AF5","AF7","AF8","T7","T8"]

eeg_concatenados = concatenateEEGs(sujeto, sesion).drop_channels(channels_to_drop,"ignore")

freqs = [50, 100, 150]  # Frecuencias a filtrar (Hz)
eeg_concatenados.notch_filter(freqs=freqs, picks='eeg', method='spectrum_fit', filter_length='auto', phase='zero')

eeg_concatenados.filter(l_freq=8, h_freq=28, 
           picks='eeg', 
           method='fir', 
           phase='zero-double', 
           fir_window='hamming',
           filter_length='auto')


## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = -3, 4
event_ids = dict(IZQUIERDA=1, DERECHA=2)
epocas_concatenadas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True)

raw_eventos = mne.events_from_annotations(eeg_concatenados, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

epocas_concatenadas.plot(scalings = 80,show=True, block=True,
                          events=eventos,
                          event_id=event_ids,
                          event_color=dict(IZQUIERDA="red", DERECHA="blue"))

## Promedio para cada tipo de clase. Esto me da un objeto Evoked
## https://mne.tools/stable/generated/mne.Evoked.html#mne.Evoked
trials_izq_avg = epocas_concatenadas["IZQUIERDA"].average()
trials_der_avg = epocas_concatenadas["DERECHA"].average()

trials_izq_avg.plot()

trials_izq_avg.plot_joint([-0.5,0,0.1,0.4,1])#_image("C3")
trials_izq_avg.plot_topomap(times=[-0.5, 0.1, 0.4], average=0.1)

plt.plot(trials_izq_avg.get_data().mean(axis=0))
plt.show()



# # 1. Carga de datos preprocesados
# raw = mne.io.read_raw_fif('ruta_al_archivo_preprocesado.fif', preload=True)

# # 2. Eventos y creación de epochs
# events, event_ids = mne.events_from_annotations(raw)

# # Define la duración del epoch para análisis ERP
# tmin, tmax = -0.2, 1.0  # Tiempo previo y posterior al cue visual
# epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=tmin, tmax=tmax,
#                     baseline=(-0.2, 0), preload=True)

# # 3. Promedio ERP (Potencial Evocado)
# erps = epochs.average()
# erps.plot_joint(title='Potenciales Evocados', times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

# # 4. Topografías para distintos momentos temporales
# times = np.linspace(0.0, 0.5, 6)
# erps.plot_topomap(times=times, title='Topografía del ERP', cmap='RdBu_r')

# # 5. Análisis espectral (Time-Frequency)
# frequencies = np.arange(1., 40., 1.)  # Rango de frecuencias para el análisis
# n_cycles = frequencies / 2.  # Ciclos por frecuencia
# power = mne.time_frequency.tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles,
#                                       use_fft=True, return_itc=False,
#                                       decim=3, n_jobs=1)

# # Visualización espectral de un canal relevante (e.g., Oz para ERPs visuales)
# power.plot(picks='Oz', title='Análisis Espectral (TFR) en Oz', cmap='RdBu_r')