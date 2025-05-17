import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs
from mne.time_frequency import tfr_morlet
from scipy import stats
import json


## 1. ******* CARGAMOS Y CONCATENAMOS LOS DATOS PARA EL/LA SUJETO EN CUESTIÓN *******
with open('codes\\parameters.json', 'r') as f:
    parameters = json.load(f)

sujeto=8
sesion=1
sfreq = 512

tipo_sesion = "Ejecutada" if sesion == 1 else "Imaginada"

channels_to_drop = parameters["channels_to_drop"]
pick = parameters["pick"]

eeg_concatenados = concatenateEEGs(sujeto, sesion, apply_ica=False).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")

l_freq, h_freq = parameters["banda_completa"]
eeg_concatenados.filter(l_freq=l_freq, h_freq=h_freq,
           picks="eeg", 
           method="fir", 
           phase="zero-double", 
           fir_window="hamming",
           filter_length="auto")


## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = -3, 4
event_ids = dict(IZQUIERDA=1, DERECHA=2)
amplitude_rejection = parameters["amplitude_rejection"]
epocas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True,reject={"eeg":amplitude_rejection})

raw_eventos = mne.events_from_annotations(eeg_concatenados, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

# epocas.plot(scalings = 80,show=True, block=True,
#                           events=eventos,
#                           event_id=event_ids,
#                           event_color=dict(IZQUIERDA="red", DERECHA="blue"))


chs_names = parameters["electrodos"]
chizq_n = chs_names[0] #nombre del canal izquierdo
chder_n = chs_names[1] #nombre del canal izquierdo
chizq_i = eeg_concatenados.ch_names.index(chs_names[0]) #índice canal izquierdo
chder_i = eeg_concatenados.ch_names.index(chs_names[1]) #índice canal derecho
cluster_izq = [eeg_concatenados.ch_names.index(ch) for ch in parameters["cluster_electrodos_izq"]]
cluster_der = [eeg_concatenados.ch_names.index(ch) for ch in parameters["cluster_electrodos_der"]]
clase_izquierda = epocas["IZQUIERDA"]
clase_derecha = epocas["DERECHA"]


## 3. ************************ ANALISIS ESPECTRAL ************************
freqs = np.arange(l_freq, h_freq, 0.1)  # Frecuencias a filtrar (Hz)
n_cycles = freqs / 1.5
baseline_rest = parameters["baseline_rest"]  # Intervalo de tiempo para el baseline
baseline_pretask = parameters["baseline_pretask"]  # Intervalo de tiempo para el baseline
baseline_task = parameters["baseline_task"]
baseline_postask = parameters["baseline_postask"]

trials_izq_rest = clase_izquierda.copy().crop(tmin=baseline_rest[0], tmax=baseline_rest[1])
trials_izq_task = clase_izquierda.copy().crop(tmin=baseline_task[0], tmax=baseline_task[1])
trials_izq_postask = clase_izquierda.copy().crop(tmin=baseline_postask[0], tmax=baseline_postask[1])
trials_der_rest = clase_derecha.copy().crop(tmin=baseline_rest[0], tmax=baseline_rest[1])
trials_der_task = clase_derecha.copy().crop(tmin=baseline_task[0], tmax=baseline_task[1])
trials_der_postask = clase_derecha.copy().crop(tmin=baseline_postask[0], tmax=baseline_postask[1])

##obtengo el psd usando multitaper
log = True
psd_izq_rest, freqline_rest = mne.time_frequency.psd_array_welch(trials_izq_rest.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_izq_task, freqline_task = mne.time_frequency.psd_array_welch(trials_izq_task.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_izq_postask, freqline_postask = mne.time_frequency.psd_array_welch(trials_izq_postask.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_der_rest, _ = mne.time_frequency.psd_array_welch(trials_der_rest.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_der_task, _ = mne.time_frequency.psd_array_welch(trials_der_task.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)
psd_der_postask, _ = mne.time_frequency.psd_array_welch(trials_der_postask.get_data(), sfreq=sfreq, fmin=l_freq, fmax=h_freq, n_jobs=1)

# log = True
# psd_izq_rest = mne.time_frequency.tfr_array_multitaper(trials_izq_rest.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
# psd_izq_task = mne.time_frequency.tfr_array_multitaper(trials_izq_task.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
# psd_izq_postask = mne.time_frequency.tfr_array_multitaper(trials_izq_postask.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
# psd_der_rest = mne.time_frequency.tfr_array_multitaper(trials_der_rest.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
# psd_der_task = mne.time_frequency.tfr_array_multitaper(trials_der_task.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
# psd_der_postask = mne.time_frequency.tfr_array_multitaper(trials_der_postask.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)

plt.plot(freqs,psd_der_postask.mean((0,3))[cluster_izq,:].mean(0))
plt.show()

if log:
    psd_izq_rest = np.log10(psd_izq_rest)
    psd_izq_task = np.log10(psd_izq_task)
    psd_izq_postask = np.log10(psd_izq_postask)
    psd_der_rest = np.log10(psd_der_rest)
    psd_der_task = np.log10(psd_der_task)
    psd_der_postask = np.log10(psd_der_postask)

psd_izq_rest_pot = psd_izq_rest.mean(axis=0)
psd_izq_task_pot = psd_izq_task.mean(axis=0)
psd_izq_postask_pot = psd_izq_postask.mean(axis=0)
psd_der_rest_pot = psd_der_rest.mean(axis=0)
psd_der_task_pot = psd_der_task.mean(axis=0)
psd_der_postask_pot = psd_der_postask.mean(axis=0)


# if no_cluster:
#     fig, axes = plt.subplots(2, 2, figsize=(14, 8))
#     axes[0, 0].plot(freqline_rest, psd_izq_rest_pot[chder_i, :], label="Rest", color=crest, linestyle='--')
#     axes[0, 0].plot(freqline_task, psd_izq_task_pot[chder_i, :], label="Tarea", color=ctask, linewidth=2)
#     axes[0, 0].plot(freqline_postask, psd_izq_postask_pot[chder_i, :], label="Post tarea", color=cposttask, linewidth=2)
#     axes[0, 0].set_title(f"IZQUIERDA {tipo_sesion} sobre {chder_i}")

#     axes[0, 1].plot(freqline_rest, psd_izq_rest_pot[chizq_i, :], label="Rest", color=crest, linestyle='--')
#     axes[0, 1].plot(freqline_task, psd_izq_task_pot[chizq_i, :], label="Tarea", color=ctask, linewidth=2)
#     axes[0, 1].plot(freqline_postask, psd_izq_postask_pot[chizq_i, :], label="Post tarea", color=cposttask, linewidth=2)
#     axes[0, 1].set_title(f"IZQUIERDA {tipo_sesion} sobre {chizq_i}")

#     axes[1, 0].plot(freqline_rest, psd_der_rest_pot[chder_i, :], label="Rest", color=crest, linestyle='--')
#     axes[1, 0].plot(freqline_task, psd_der_task_pot[chder_i, :], label="Tarea", color=ctask, linewidth=2)
#     axes[1, 0].plot(freqline_postask, psd_izq_postask_pot[chder_i, :], label="Post tarea", color=cposttask, linewidth=2)
#     axes[1, 0].set_title(f"DERECHA {tipo_sesion} sobre {chder_i}")

#     axes[1, 1].plot(freqline_rest, psd_der_rest_pot[chizq_i, :], label="Rest", color=crest, linestyle='--')
#     axes[1, 1].plot(freqline_task, psd_der_task_pot[chizq_i, :], label="Tarea", color=ctask, linewidth=2)
#     axes[1, 1].plot(freqline_postask, psd_der_postask_pot[chizq_i, :], label="Post Tarea", color=cposttask, linewidth=2)
#     axes[1, 1].set_title(f"DERECHA {tipo_sesion} sobre {chizq_i}")
# else:
#     fig, axes = plt.subplots(2, 2, figsize=(14, 8))
#     axes[0, 0].plot(freqline_rest, psd_izq_rest_pot[cluster_der, :].mean(0), label="Rest", color=crest, linestyle='--')
#     axes[0, 0].plot(freqline_task, psd_izq_task_pot[cluster_der, :].mean(0), label="Tarea", color=ctask, linewidth=2)
#     axes[0, 0].plot(freqline_postask, psd_izq_postask_pot[cluster_der, :].mean(0), label="Post tarea", color=cposttask, linewidth=2)
#     axes[0, 0].set_title(f"IZQUIERDA {tipo_sesion} promedio sobre {parameters["cluster_electrodos_der"]}")

#     axes[0, 1].plot(freqline_rest, psd_izq_rest_pot[cluster_izq, :].mean(0), label="Rest", color=crest, linestyle='--')
#     axes[0, 1].plot(freqline_task, psd_izq_task_pot[cluster_izq, :].mean(0), label="Tarea", color=ctask, linewidth=2)
#     axes[0, 1].plot(freqline_postask, psd_izq_postask_pot[cluster_izq, :].mean(0), label="Post tarea", color=cposttask, linewidth=2)
#     axes[0, 1].set_title(f"IZQUIERDA {tipo_sesion} promedio sobre {parameters["cluster_electrodos_izq"]}")

#     axes[1, 0].plot(freqline_rest, psd_der_rest_pot[cluster_der, :].mean(0), label="Rest", color=crest, linestyle='--')
#     axes[1, 0].plot(freqline_task, psd_der_task_pot[cluster_der, :].mean(0), label="Tarea", color=ctask, linewidth=2)
#     axes[1, 0].plot(freqline_postask, psd_izq_postask_pot[cluster_der, :].mean(0), label="Post tarea", color=cposttask, linewidth=2)
#     axes[1, 0].set_title(f"DERECHA {tipo_sesion} promedio sobre {parameters["cluster_electrodos_der"]}")

#     axes[1, 1].plot(freqline_rest, psd_der_rest_pot[cluster_izq, :].mean(0), label="Rest", color=crest, linestyle='--')
#     axes[1, 1].plot(freqline_task, psd_der_task_pot[cluster_izq, :].mean(0), label="Tarea", color=ctask, linewidth=2)
#     axes[1, 1].plot(freqline_postask, psd_der_postask_pot[cluster_izq, :].mean(0), label="Post Tarea", color=cposttask, linewidth=2)
#     axes[1, 1].set_title(f"DERECHA {tipo_sesion} promedio sobre {parameters["cluster_electrodos_izq"]}")

# for ax in axes.flat:
#     ax.set_xlabel("Frecuencia (Hz)")
#     ax.set_ylabel("Potencia $uV^2$")
#     ax.axhline(0, color='grey', linestyle='--')
#     ax.axvline(l_freq+1, color='grey', linestyle='--')
#     ax.legend()
#     ax.grid()
# plt.tight_layout()
# plt.show()


## ************** reactive band selection ********************
# Umbral para significancia estadística

no_cluster = False

if no_cluster:
    reactive_psd_izq_rest = psd_izq_rest[:,chder_i,:].mean(0)
    reactive_psd_izq_task = psd_izq_task[:,chder_i,:].mean(0)
    reactive_psd_izq_postask = psd_izq_postask[:,chder_i,:].mean(0)

    reactive_psd_der_rest = psd_der_rest[:,chizq_i,:].mean(0)
    reactive_psd_der_task = psd_der_task[:,chizq_i,:].mean(0)
    reactive_psd_der_postask = psd_der_postask[:,chizq_i,:].mean(0)

else:
    reactive_psd_izq_rest = psd_izq_rest[:,cluster_der,:].mean((0,1))
    reactive_psd_izq_task = psd_izq_task[:,cluster_der,:].mean((0,1))
    reactive_psd_izq_postask = psd_izq_postask[:,cluster_der,:].mean((0,1))

    reactive_psd_der_rest = psd_der_rest[:,cluster_izq,:].mean((0,1))
    reactive_psd_der_task = psd_der_task[:,cluster_izq,:].mean((0,1))
    reactive_psd_der_postask = psd_der_postask[:,cluster_izq,:].mean((0,1))

confidence = 0.95
n = reactive_psd_izq_task.shape[0] #n es el mismo para todas las variables a usar
t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df = n - 1)

#cálculo de diferencias, media e intervalo de confianza para clase izquierda sobre electrodos de la derecha
diff_tr_izq = reactive_psd_izq_task - reactive_psd_izq_rest #diferencia de potencia entre task y rest clase izquierda
mean_diff_tr_izq = np.mean(diff_tr_izq)
std_tr_izq = np.std(diff_tr_izq, ddof=1) / np.sqrt(n) #error estándar de la media (debemos dividir por n^(1/2))
confinter_tr_izq = 1.96 * std_tr_izq #intervalo de confianza para task - rest

diff_pr_izq = reactive_psd_izq_postask - reactive_psd_izq_rest #diferencia de potencia entre post_task y rest clase izquierda
mean_diff_pr_izq = np.mean(diff_pr_izq)
std_pr_izq = np.std(diff_pr_izq, ddof=1) / np.sqrt(n) #error estándar de la media (debemos dividir por n^(1/2))
confinter_pr_izq = 1.96 * std_pr_izq #intervalo de confianza para post_task - rest

##repito para derecha
diff_tr_der = reactive_psd_der_task - reactive_psd_der_rest #diferencia de potencia entre task y rest clase derecha
mean_diff_tr_der = np.mean(diff_tr_der)
std_tr_der = np.std(diff_tr_der) / np.sqrt(n)
confinter_tr_der = 1.96 * std_tr_der

diff_pr_der = reactive_psd_der_postask - reactive_psd_der_rest #diferencia de potencia entre posta_task y rest clase derecha
mean_diff_pr_der = np.mean(diff_pr_der)
std_pr_der = np.std(diff_pr_der) / np.sqrt(n)
confinter_pr_der = 1.96 * std_pr_der

##vamos a intentar replicar la gráfica derecha y arriba de 45.2 del artículo 
##https://neupsykey.com/eeg-event-related-desynchronization-erd-and-event-related-synchronization-ers/#R1-45

fig, axes = plt.subplots(4, 1, figsize=(6, 8), constrained_layout=True)
crest, ctask, cposttask = "#5dade2", "#45b27b", "#e74c3c"
cdiff = "#fc9403"
if True:
    axes[0].plot(freqline_rest, diff_tr_izq, label=r"${\Delta}{{\mu}^2}$", color=cdiff)
    axes[0].axhline(mean_diff_tr_izq, color='grey')
    axes[0].axhline(mean_diff_tr_izq + confinter_tr_izq, color='grey', linestyle='-.')
    axes[0].axhline(mean_diff_tr_izq - confinter_tr_izq, color='grey', linestyle='-.')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].legend(loc="upper right")
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axes[0].set_title(f"A1: {baseline_task} segundos", loc='left')

    axes[1].plot(freqline_rest, psd_izq_rest_pot[chder_i, :], label="R", linestyle=':', linewidth=2)
    axes[1].plot(freqline_task, psd_izq_task_pot[chder_i, :], label="A1", color=ctask, linewidth=2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axes[1].legend(loc="upper right")
    
    axes[2].set_title(f"A2: {baseline_postask} segundos", loc='left')
    axes[2].plot(freqline_rest, diff_pr_izq, label=r"${\Delta}{{\mu}^2}$", color=cdiff)
    axes[2].axhline(mean_diff_pr_izq, color='grey')
    axes[2].axhline(mean_diff_pr_izq + confinter_tr_izq, color='grey', linestyle='-.')
    axes[2].axhline(mean_diff_pr_izq - confinter_tr_izq, color='grey', linestyle='-.')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['bottom'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].legend(loc="upper right")
    axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    axes[3].plot(freqline_rest, psd_izq_rest_pot[chder_i, :], label="R", linestyle=':', linewidth=2)
    axes[3].plot(freqline_postask, psd_izq_postask_pot[chder_i, :], label="A2", color=cposttask, linewidth=2)
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    axes[3].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axes[3].legend(loc="upper right")

plt.suptitle(f"")
# plt.tight_layout()
plt.show()


# color="#3f89d3"
# fig, axes = plt.subplots(2, 2, figsize=(16, 8))
# axes[0, 0].plot(freqline_rest, diff_tr_izq, color=color)
# axes[0, 0].axhline(mean_diff_tr_izq, color='grey', label='Media')
# axes[0, 0].axhline(mean_diff_tr_izq + conf_interval_cuebase_izq, color='grey', linestyle='-.')
# axes[0, 0].axhline(mean_diff_tr_izq - conf_interval_cuebase_izq, color='grey', linestyle='-.')
# axes[0, 0].set_title("Diferencia Baseline y Cue IZQUIERDA - Electrodo C4", fontsize=12)
# axes[0, 1].plot(freqline_rest, diff_freqs_cuepostask_izq, color=color)
# axes[0, 1].axhline(mean_diff_cuepostask_izq, color='grey', label='Media')
# axes[0, 1].axhline(mean_diff_cuepostask_izq + conf_interval_cuepostask_izq, color='grey', linestyle='-.')
# axes[0, 1].axhline(mean_diff_cuepostask_izq - conf_interval_cuepostask_izq, color='grey', linestyle='-.')
# axes[0, 1].set_title("Diferencia Baseline y Post Tarea IZQUIERDA - Electrodo C4", fontsize=12)
# axes[1, 0].plot(freqline_rest, diff_tr_der, color=color)
# axes[1, 0].axhline(mean_diff_tr_der, color='grey', label='Media')
# axes[1, 0].axhline(mean_diff_tr_der + conf_interval_cuebase_der, color='grey', linestyle='-.')
# axes[1, 0].axhline(mean_diff_tr_der - conf_interval_cuebase_der, color='grey', linestyle='-.')
# axes[1, 0].set_title("Diferencia Baseline y Cue DERECHA - Electrodo C3", fontsize=12)
# axes[1, 1].plot(freqline_rest, diff_freqs_cuepostask_der, color=color)
# axes[1, 1].axhline(mean_diff_cuepostask_der, color='grey', label='Media')
# axes[1, 1].axhline(mean_diff_cuepostask_der + conf_interval_cuepostask_der, color='grey', linestyle='-.')
# axes[1, 1].axhline(mean_diff_cuepostask_der - conf_interval_cuepostask_der, color='grey', linestyle='-.')
# axes[1, 1].set_title("Diferencia Baseline y Post Tarea DERECHA - Electrodo C3", fontsize=12)
# for ax in axes.flat:
#     ax.set_xlabel("Frecuencia (Hz)")
#     ax.set_ylabel("($uV^2$)")
#     ax.legend()
#     ax.grid()
# plt.tight_layout()
# plt.show()