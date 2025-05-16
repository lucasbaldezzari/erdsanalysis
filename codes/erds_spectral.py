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

# epocas.plot(scalings = 80,show=True, block=True,
#                           events=eventos,
#                           event_id=event_ids,
#                           event_color=dict(IZQUIERDA="red", DERECHA="blue"))


electrodos = ["C3","C4"]
c3_index = eeg_concatenados.ch_names.index(electrodos[0])
c4_index = eeg_concatenados.ch_names.index(electrodos[1])
clase_izquierda = epocas["IZQUIERDA"]
clase_derecha = epocas["DERECHA"]
clase_izquierda_avg = clase_izquierda.average()
clase_derecha_avg= clase_derecha.average()


## 3. ************************ ANALISIS ESPECTRAL ************************
freqs = np.arange(l_freq, h_freq, 0.5)  # Frecuencias a filtrar (Hz)
baseline_rest = (-1.5, -0.5)  # Intervalo de tiempo para el baseline
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
psd_izq_rest_pot = (psd_izq_rest).mean(axis=0)
psd_izq_task_pot = (psd_izq_task).mean(axis=0)
psd_izq_postask_pot = (psd_izq_postask).mean(axis=0)
psd_der_rest_pot = (psd_der_rest).mean(axis=0)
psd_der_task_pot = (psd_der_task).mean(axis=0)
psd_der_postask_pot = (psd_der_postask).mean(axis=0)


crest, ctask, cposttask ="#5dade2", "#45b27b", "#e74c3c"
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].plot(freqline_rest, psd_izq_rest_pot[c4_index, :], label="Rest", color=crest, linestyle='--')
axes[0, 0].plot(freqline_task, psd_izq_task_pot[c4_index, :], label="Tarea", color=ctask, linewidth=2)
axes[0, 0].plot(freqline_postask, psd_izq_postask_pot[c4_index, :], label="Post tarea", color=cposttask, linewidth=2)
axes[0, 0].set_title(f"IZQUIERDA {tipo_sesion} sobre {electrodos[1]}")

axes[0, 1].plot(freqline_rest, psd_izq_rest_pot[c3_index, :], label="Rest", color=crest, linestyle='--')
axes[0, 1].plot(freqline_task, psd_izq_task_pot[c3_index, :], label="Tarea", color=ctask, linewidth=2)
axes[0, 1].plot(freqline_postask, psd_izq_postask_pot[c3_index, :], label="Post tarea", color=cposttask, linewidth=2)
axes[0, 1].set_title(f"IZQUIERDA {tipo_sesion} sobre {electrodos[0]}")

axes[1, 0].plot(freqline_rest, psd_der_rest_pot[c4_index, :], label="Rest", color=crest, linestyle='--')
axes[1, 0].plot(freqline_task, psd_der_task_pot[c4_index, :], label="Tarea", color=ctask, linewidth=2)
axes[1, 0].plot(freqline_postask, psd_izq_postask_pot[c4_index, :], label="Post tarea", color=cposttask, linewidth=2)
axes[1, 0].set_title(f"DERECHA {tipo_sesion} sobre {electrodos[0]}")

axes[1, 1].plot(freqline_rest, psd_der_rest_pot[c3_index, :], label="Rest", color=crest, linestyle='--')
axes[1, 1].plot(freqline_task, psd_der_task_pot[c3_index, :], label="Tarea", color=ctask, linewidth=2)
axes[1, 1].plot(freqline_postask, psd_der_postask_pot[c3_index, :], label="Post Tarea", color=cposttask, linewidth=2)
axes[1, 1].set_title(f"DERECHA {tipo_sesion} sobre {electrodos[1]}")

for ax in axes.flat:
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Potencia $uV^2$")
    ax.axhline(0, color='grey', linestyle='--')
    ax.axvline(l_freq+1, color='grey', linestyle='--')
    ax.legend()
    ax.grid()
plt.tight_layout()
plt.show()


##reactive band selection
# Umbral para significancia estadística
diff_freqs_cuebase_izq = (psd_izq_task)[:,c4_index,:].mean(axis=0) - (psd_izq_rest)[:,c4_index,:].mean(axis=0)
mean_diff_cuebase_izq = np.mean(diff_freqs_cuebase_izq)
std_cuebase_izq = np.std(diff_freqs_cuebase_izq) / np.sqrt(diff_freqs_cuebase_izq.shape[0])
conf_interval_cuebase_izq = 1.96 * std_cuebase_izq
##repito para cuebase vs posttask izq
diff_freqs_cuepostask_izq = (psd_izq_postask)[:,c4_index,:].mean(axis=0) - (psd_izq_rest)[:,c4_index,:].mean(axis=0)
mean_diff_cuepostask_izq = np.mean(diff_freqs_cuepostask_izq)
std_cuepostask_izq = np.std(diff_freqs_cuepostask_izq) / np.sqrt(diff_freqs_cuepostask_izq.shape[0])
conf_interval_cuepostask_izq = 1.96 * std_cuepostask_izq
##repito para derecha
diff_freqs_cuebase_der = (psd_der_task)[:,c3_index,:].mean(axis=0) - (psd_der_rest)[:,c3_index,:].mean(axis=0)
mean_diff_cuebase_der = np.mean(diff_freqs_cuebase_der)
std_cuebase_der = np.std(diff_freqs_cuebase_der) / np.sqrt(diff_freqs_cuebase_der.shape[0])
conf_interval_cuebase_der = 1.96 * std_cuebase_der
##repito para cuebase vs posttask der
diff_freqs_cuepostask_der = (psd_der_postask)[:,c3_index,:].mean(axis=0) - (psd_der_task)[:,c3_index,:].mean(axis=0)
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
    ax.set_ylabel("($uV^2$)")
    ax.legend()
    ax.grid()
plt.tight_layout()
plt.show()