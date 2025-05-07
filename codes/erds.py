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
from scipy.stats import ttest_rel


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
baseline = Baseline((-1.5, -0.5))  # Intervalo de tiempo para el baseline
erds_izq = getHilbertERDS(clase_izquierda, baseline, apply_smooth=True, window_smoothing=50)
erds_der = getHilbertERDS(clase_derecha, baseline, apply_smooth=True, window_smoothing=50)

plt.style.use('default')
plt.style.available

plt.figure(figsize=(10, 5))
plt.plot(clase_izquierda.times, erds_izq[c3_index, :], label=f'ERDS% en C3')
plt.plot(clase_izquierda.times, erds_izq[c4_index, :], label=f'ERDS% en C4')
plt.axvline(0, color='k', linestyle='--', label='Cue onset')
plt.axhline(0, color='grey', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('ERDS (%)')
plt.title(f'Curva EDRS% clase IZQUIERDA')
plt.legend()
plt.xlim(-1, 3)
plt.ylim(-80, 80)
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(clase_derecha.times, erds_der[c3_index, :], label=f'ERDS% en C3')
plt.plot(clase_derecha.times, erds_der[c4_index, :], label=f'ERDS% en C4')
plt.axvline(0, color='k', linestyle='--', label='Cue onset')
plt.axhline(0, color='grey', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('ERDS (%)')
plt.title(f'Curva EDRS% clase DERECHA')
plt.legend()
plt.xlim(-1, 3)
plt.ylim(-80, 80)
plt.grid()
plt.show()

## 4. ************************ ANALISIS ESPECTRAL ************************
freqs = np.arange(l_freq, h_freq, 0.5)  # Frecuencias a filtrar (Hz)
baseline_rest = (-2, -1)  # Intervalo de tiempo para el baseline
baseline_pretask = (-0.5, 0)  # Intervalo de tiempo para el baseline
baseline_task = (0, 1)
baseline_postask = (2, 3)

trials_izq_rest = clase_izquierda.copy().crop(tmin=baseline_rest[0], tmax=baseline_rest[1])
trials_izq_task = clase_izquierda.copy().crop(tmin=baseline_task[0], tmax=baseline_task[1])
trials_izq_postask = clase_izquierda.copy().crop(tmin=baseline_postask[0], tmax=baseline_postask[1])
trials_der_rest = clase_derecha.copy().crop(tmin=baseline_rest[0], tmax=baseline_rest[1])
trials_der_task = clase_derecha.copy().crop(tmin=baseline_task[0], tmax=baseline_task[1])
trials_der_postask = clase_derecha.copy().crop(tmin=baseline_postask[0], tmax=baseline_postask[1])

##obtengo el psd usando multitaper
psd_izq_rest, freqline_rest = mne.time_frequency.psd_array_multitaper(trials_izq_rest.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_izq_task, freqline_task = mne.time_frequency.psd_array_multitaper(trials_izq_task.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_izq_postask, freqline_postask = mne.time_frequency.psd_array_multitaper(trials_izq_postask.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_der_rest, _ = mne.time_frequency.psd_array_multitaper(trials_der_rest.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_der_task, _ = mne.time_frequency.psd_array_multitaper(trials_der_task.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_der_postask, _ = mne.time_frequency.psd_array_multitaper(trials_der_postask.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)

##convertimos a db
psd_izq_rest_db = 10 * np.log10(psd_izq_rest).mean(axis=0)
psd_izq_task_db = 10 * np.log10(psd_izq_task).mean(axis=0)
psd_izq_postask_db = 10 * np.log10(psd_izq_postask).mean(axis=0)
psd_der_rest_db = 10 * np.log10(psd_der_rest).mean(axis=0)
psd_der_task_db = 10 * np.log10(psd_der_task).mean(axis=0)
psd_der_postask_db = 10 * np.log10(psd_der_postask).mean(axis=0)

crest, ctask, cposttask ="#5dade2", "#45b27b", "#e74c3c"
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].plot(freqline_rest, psd_izq_rest_db[c4_index, :], label="Baseline", color=crest, linestyle='--')
axes[0, 0].plot(freqline_task, psd_izq_task_db[c4_index, :], label="Tarea", color=ctask, linewidth=2)
# Líneas de intervalo de confianza
axes[0, 0].set_title("Baseline y Cue IZQUIERDA - C4")
axes[0, 1].plot(freqline_rest, psd_izq_rest_db[c4_index, :], label="Baseline", color=crest, linestyle='--')
axes[0, 1].plot(freqline_postask, psd_izq_postask_db[c4_index, :], label="Tarea", color=cposttask, linewidth=2)
axes[0, 1].set_title("Baseline y Post Tarea IZQUIERDA - C4")
axes[1, 0].plot(freqline_rest, psd_der_rest_db[c3_index, :], label="Baseline", color=crest, linestyle='--')
axes[1, 0].plot(freqline_task, psd_der_task_db[c3_index, :], label="Tarea", color=ctask, linewidth=2)
axes[1, 0].set_title("Baseline y Cue DERECHA - C3")
axes[1, 1].plot(freqline_rest, psd_der_rest_db[c3_index, :], label="Baseline", color=crest, linestyle='--')
axes[1, 1].plot(freqline_postask, psd_der_postask_db[c3_index, :], label="Post Tarea", color=cposttask, linewidth=2)
axes[1, 1].set_title("Baseline y Post Tarea DERECHA - C3")
for ax in axes.flat:
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Potencia (dB)")
    ax.axhline(0, color='grey', linestyle='--')
    ax.axvline(l_freq+1, color='grey', linestyle='--')
    ax.legend()
    ax.grid()
plt.tight_layout()
plt.show()

##reactive band selection
# Umbral para significancia estadística
diff_freqs_cuebase_izq = np.log10(psd_izq_task)[:,c4_index,:].mean(axis=0) - np.log10(psd_izq_rest)[:,c4_index,:].mean(axis=0)
mean_diff_cuebase_izq = np.mean(diff_freqs_cuebase_izq)
std_cuebase_izq = np.std(diff_freqs_cuebase_izq) / np.sqrt(diff_freqs_cuebase_izq.shape[0])
conf_interval_cuebase_izq = 1.96 * std_cuebase_izq
##repito para cuebase vs posttask izq
diff_freqs_cuepostask_izq = np.log10(psd_izq_postask)[:,c4_index,:].mean(axis=0) - np.log10(psd_izq_rest)[:,c4_index,:].mean(axis=0)
mean_diff_cuepostask_izq = np.mean(diff_freqs_cuepostask_izq)
std_cuepostask_izq = np.std(diff_freqs_cuepostask_izq) / np.sqrt(diff_freqs_cuepostask_izq.shape[0])
conf_interval_cuepostask_izq = 1.96 * std_cuepostask_izq
##repito para derecha
diff_freqs_cuebase_der = np.log10(psd_der_task)[:,c3_index,:].mean(axis=0) - np.log10(psd_der_rest)[:,c3_index,:].mean(axis=0)
mean_diff_cuebase_der = np.mean(diff_freqs_cuebase_der)
std_cuebase_der = np.std(diff_freqs_cuebase_der) / np.sqrt(diff_freqs_cuebase_der.shape[0])
conf_interval_cuebase_der = 1.96 * std_cuebase_der
##repito para cuebase vs posttask der
diff_freqs_cuepostask_der = np.log10(psd_der_postask)[:,c3_index,:].mean(axis=0) - np.log10(psd_der_task)[:,c3_index,:].mean(axis=0)
mean_diff_cuepostask_der = np.mean(diff_freqs_cuepostask_der)
std_cuepostask_der = np.std(diff_freqs_cuepostask_der) / np.sqrt(diff_freqs_cuepostask_der.shape[0])
conf_interval_cuepostask_der = 1.96 * std_cuepostask_der

color="#3f89d3"
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes[0, 0].plot(freqline_rest, diff_freqs_cuebase_izq, color=color)
axes[0, 0].axhline(mean_diff_cuebase_izq, color='grey', label='Media')
axes[0, 0].axhline(mean_diff_cuebase_izq + conf_interval_cuebase_izq, color='grey', linestyle='-.')
axes[0, 0].axhline(mean_diff_cuebase_izq - conf_interval_cuebase_izq, color='grey', linestyle='-.')
axes[0, 0].set_title("Diferencia Baseline y Cue IZQUIERDA - Electrodo C4", fontsize=12)
axes[0, 1].plot(freqline_rest, diff_freqs_cuepostask_izq, color=color)
axes[0, 1].axhline(mean_diff_cuepostask_izq, color='grey', label='Media')
axes[0, 1].axhline(mean_diff_cuepostask_izq + conf_interval_cuepostask_izq, color='grey', linestyle='-.')
axes[0, 1].axhline(mean_diff_cuepostask_izq - conf_interval_cuepostask_izq, color='grey', linestyle='-.')
axes[0, 1].set_title("Diferencia Baseline y Post Tarea IZQUIERDA - Electrodo C4", fontsize=12)
axes[1, 0].plot(freqline_rest, diff_freqs_cuebase_der, color=color)
axes[1, 0].axhline(mean_diff_cuebase_der, color='grey', label='Media')
axes[1, 0].axhline(mean_diff_cuebase_der + conf_interval_cuebase_der, color='grey', linestyle='-.')
axes[1, 0].axhline(mean_diff_cuebase_der - conf_interval_cuebase_der, color='grey', linestyle='-.')
axes[1, 0].set_title("Diferencia Baseline y Cue DERECHA - Electrodo C3", fontsize=12)
axes[1, 1].plot(freqline_rest, diff_freqs_cuepostask_der, color=color)
axes[1, 1].axhline(mean_diff_cuepostask_der, color='grey', label='Media')
axes[1, 1].axhline(mean_diff_cuepostask_der + conf_interval_cuepostask_der, color='grey', linestyle='-.')
axes[1, 1].axhline(mean_diff_cuepostask_der - conf_interval_cuepostask_der, color='grey', linestyle='-.')
axes[1, 1].set_title("Diferencia Baseline y Post Tarea DERECHA - Electrodo C3", fontsize=12)
for ax in axes.flat:
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Diferencia (dB)")
    ax.legend()
    ax.grid()
plt.tight_layout()
plt.show()

## 5. ************************ ANALISIS TIEMPO-FRECUENCIA ************************
### Aplicar el análisis de Morlet para obtener la potencia en el rango de frecuencias deseado y luego tomar los datos, aplicar el baseline usando el 
### los datos del tiempo baseline y así rasignar la data al objeto MNE.

tfr_izq = tfr_morlet(clase_izquierda, freqs=freqs, n_cycles=freqs/2., return_itc=False, average=False)
tfr_der = tfr_morlet(clase_derecha, freqs=freqs, n_cycles=freqs/2., return_itc=False, average=False)

#indices donde freqs sea mayor a 10 y menor a 12
indices = np.where((freqs >= 8) & (freqs <= 13))[0]

tfr_izq_data_1012 = tfr_izq.data[:, :, indices, :].mean(axis=2) ##media en el eje de frecuencias
##aplico baseline para cada trial
for trial in range(tfr_izq_data_1012.shape[0]):
    baseline_mean = baseline.apply(tfr_izq_data_1012[trial, :], tfr_izq.times)
    tfr_izq_data_1012[trial, :] = 100*(tfr_izq_data_1012[trial, :] - baseline_mean) / baseline_mean
tfr_izq_data_1012_c4 = tfr_izq_data_1012.mean(axis=0)[c4_index, :]
##repito para derecha
tfr_der_data_1012 = tfr_der.data[:, :, indices, :].mean(axis=2) ##media en el eje de frecuencias
for trial in range(tfr_der_data_1012.shape[0]):
    baseline_mean = baseline.apply(tfr_der_data_1012[trial, :], tfr_der.times)
    tfr_der_data_1012[trial, :] = 100*(tfr_der_data_1012[trial, :] - baseline_mean) / baseline_mean
tfr_der_data_1012_c3 = tfr_der_data_1012.mean(axis=0)[c3_index, :]

window_size = 512
tfr_izq_data_1012_c4 = np.convolve(tfr_izq_data_1012_c4, np.ones(window_size)/window_size, mode='same')
tfr_der_data_1012_c3 = np.convolve(tfr_der_data_1012_c3, np.ones(window_size)/window_size, mode='same')
plt.plot(tfr_izq.times, tfr_izq_data_1012_c4, label="IZQUIERDA - C4", color="blue")
plt.plot(tfr_der.times, tfr_der_data_1012_c3, label="DERECHA - C3", color="red")
plt.legend()
plt.show()
