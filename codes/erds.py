# Código para analizar ERPs visuales usando MNE-Python
# https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html#sphx-glr-auto-tutorials-evoked-30-eeg-erp-py

import mne
import numpy as np
import matplotlib.pyplot as plt
from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
from codes.utils import concatenateEEGs, loadOA, getHilbertERDS
from mne.time_frequency import tfr_morlet
from codes.utils import Baseline

## 1. ******* Cargamos y concatenamos los datos para el sujeto y la sesión en cuestión *******
sujeto=8
sesion=2
sfreq = 512


channels_to_drop = ["FP1","FP2","FPz","Fz","F8","F7","AF3","AF4","AF5","AF7","AF8","T7","T8","F9","F10"]
pick = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6",]

eeg_concatenados = concatenateEEGs(sujeto, sesion, apply_ica=False).pick(pick,"ignore")
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
                                 baseline=None, preload=True)

raw_eventos = mne.events_from_annotations(eeg_concatenados, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

# epocas_concatenadas.plot(scalings = 80,show=True, block=True,
#                           events=eventos,
#                           event_id=event_ids,
#                           event_color=dict(IZQUIERDA="red", DERECHA="blue"))

##separo los datos de cada clase
c3_index, c4_index = eeg_concatenados.ch_names.index("C3"), eeg_concatenados.ch_names.index("C4")
clase_izquierda = epocas_concatenadas["IZQUIERDA"]
clase_derecha = epocas_concatenadas["DERECHA"]
clase_izquierda_average = clase_izquierda.average()
clase_derecha_average = clase_derecha.average()

## Pasos a seguir

#1. (OK ESTE PUNTO) Calcular curvas ERDS% para cada clase y para algunos canales de interés sobre
# los trials promediados, sobre todo los que están en la región motora. Usar el registro de ojos abiertos y descansados como linea base.
# También se podría usar Hilbert para esto. (OK ESTE PUNTO)

#2. Repetir paso 1 pero ahora habiendo aplicado filtro laplaciano a los canales C3 y C4, por ejemplo
#3. Repetir paso 1 pero ahora habiendo aplicado un CSP a los datos.
#4. Con el CSP graficar mapas topográficos para las componentes y ver que se tiene.
#5. Aplicar Time-Frequency Analysis a los datos y graficar mapas topográficos para las frecuencias de interés. Utilizar funciones propias y de MNE para comparar.

## 3. ************************ CURVAS ERDS% USANDO HILBERT ************************
# Graficar curva ERDS para un canal específico (por ejemplo, C3)
baseline = Baseline((-1.5, -0.6))  # Intervalo de tiempo para el baseline
erds_izq = getHilbertERDS(clase_izquierda, baseline, apply_smooth=True, window_smoothing=50)
erds_der = getHilbertERDS(clase_derecha, baseline, apply_smooth=True, window_smoothing=50)

plt.figure(figsize=(10, 5))
plt.plot(clase_izquierda.times, erds_izq[c3_index, :], label=f'ERDS% en C3', color ="red")
plt.plot(clase_izquierda.times, erds_izq[c4_index, :], label=f'ERDS% en C4', color ="blue")
plt.axvline(0, color='k', linestyle='--', label='Cue onset')
plt.axhline(0, color='grey', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('ERDS (%)')
plt.title(f'Curva EDRS% clase IZQUIERDA')
plt.legend()
plt.xlim(-1, 3)
plt.ylim(-100, 100)
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(clase_derecha.times, erds_der[c3_index, :], label=f'ERDS% en C3', color ="red")
plt.plot(clase_derecha.times, erds_der[c4_index, :], label=f'ERDS% en C4', color ="blue")
plt.axvline(0, color='k', linestyle='--', label='Cue onset')
plt.axhline(0, color='grey', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('ERDS (%)')
plt.title(f'Curva EDRS% clase DERECHA')
plt.legend()
plt.xlim(-1, 3)
plt.ylim(-100, 100)
plt.grid()
plt.show()

## 4. ************************ ANALISIS ESPECTRAL ************************
freqs = np.arange(l_freq-2, h_freq+2, 0.1)  # Frecuencias a filtrar (Hz)
baseline_rest = (-1.5, -0.6)  # Intervalo de tiempo para el baseline
baseline_task = (0, 1)

trials_izq_rest = clase_izquierda.copy().crop(tmin=baseline_rest[0], tmax=baseline_rest[1])
trials_izq_task = clase_izquierda.copy().crop(tmin=baseline_task[0], tmax=baseline_task[1])
trials_der_rest = clase_derecha.copy().crop(tmin=baseline_rest[0], tmax=baseline_rest[1])
trials_der_task = clase_derecha.copy().crop(tmin=baseline_task[0], tmax=baseline_task[1])

##obtengo el psd usando multitaper
psd_izq_rest, freq_line_rest = mne.time_frequency.psd_array_multitaper(trials_izq_rest.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_izq_task, freq_line_task = mne.time_frequency.psd_array_multitaper(trials_izq_task.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_der_rest, freq_line_rest = mne.time_frequency.psd_array_multitaper(trials_der_rest.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_der_task, freq_line_task = mne.time_frequency.psd_array_multitaper(trials_der_task.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)

##convertimos a db
psd_izq_rest_db = 10 * np.log10(psd_izq_rest).mean(axis=0)
psd_izq_task_db = 10 * np.log10(psd_izq_task).mean(axis=0)
psd_der_rest_db = 10 * np.log10(psd_der_rest).mean(axis=0)
psd_der_task_db = 10 * np.log10(psd_der_task).mean(axis=0)

plt.plot(freq_line_rest,psd_der_rest_db[c3_index, :],label="Izquierda Baseline")
plt.plot(freq_line_task,psd_der_task_db[c3_index, :],label="Izquierda Tarea")
plt.legend()
plt.title("C3")
plt.show()

## 5. ************************ ANALISIS TIEMPO-FRECUENCIA ************************
### Aplicar el análisis de Morlet para obtener la potencia en el rango de frecuencias deseado y luego tomar los datos, aplicar el baseline usando el 
### los datos del tiempo baseline y así rasignar la data al objeto MNE.


# ## Análisis en Tiempo-Frecuencia
# power_left = mne.time_frequency.tfr_morlet(clase_izquierda, freqs=freqs, n_cycles=freqs/2., return_itc=False, average=True)
# power_right = mne.time_frequency.tfr_morlet(clase_derecha, freqs=freqs, n_cycles=freqs/2., return_itc=False, average=True)


# power_right.plot(picks=["C1","C2"],fmin=5, fmax=36, cmap="jet")
# power_left.plot_topomap(tmin=0,tmax=1,fmin=7, baseline=(-1.4,-1.2), mode="percent", cmap="jet")

# ## 3. ************************ CURVAS ERDS SOBRE SEÑAL FILTRADA ************************

# times=epocas_concatenadas.times
# tinit, tfinal = 0, 2
# trial = 10
# colors_rect = ['#9ecfcf', '#ffc899']
# fig, ax = plt.subplots(2, 1, figsize=(12, 6))
# ax[0].plot(times, clase_izquierda.get_data()[trial-1,c4_index,:],color="grey",label=f"Izquierda Trial {trial}")
# ax[0].plot(times, clase_izquierda_average.get_data()[c3_index],color="black",label="Izquierda promedio")
# ax[1].plot(times, clase_derecha.get_data()[trial-1,c3_index,:],color="grey",label=f"Derecha Trial {trial}")
# ax[1].plot(times, clase_derecha_average.get_data()[c3_index],color="black",label="Derecha promedio")
# ax[0].set_title("C4")
# ax[1].set_title("C3")
# ax[0].set_ylabel("Amplitud (uV)")
# ax[1].set_ylabel("Amplitud (uV)")
# ax[0].axhline(0, color='#777777', linestyle="--")
# ax[1].axhline(0, color='#777777', linestyle="--")
# ax[0].axvspan(tinit,tfinal, color=colors_rect[0], alpha=0.5, label="Ventana cue IZQUIERDA")
# ax[1].axvspan(tinit,tfinal, color=colors_rect[1], alpha=0.5, label="Ventana cue DERECHA")
# fig.suptitle("ERDS %", fontsize=20)
# ax[0].legend(loc=4)
# ax[1].legend(loc=4)
# plt.show()

# times=epocas_concatenadas.times
# tinit, tfinal = 0, 2
# trial = 15
# colors_rect = ['#9ecfcf', '#ffc899']
# fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# ax.plot(times, clase_izquierda_average.get_data()[c3_index],color="red",label=f"Izquierda trial {trial}")
# ax.plot(times, clase_derecha_average.get_data()[c3_index],color="blue",label=f"Derecha trial {trial}")
# ax.set_ylabel("Amplitud (uV)")
# ax.axhline(0, color='#777777', linestyle="--")
# ax.axvspan(tinit,tfinal, color="grey", alpha=0.5, label="Ventana cue")
# fig.suptitle(f"C4 para trial {trial}", fontsize=16)
# ax.legend(loc=4)
# plt.show()