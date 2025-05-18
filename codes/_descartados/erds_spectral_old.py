import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs
from mne.time_frequency import tfr_morlet
from scipy import stats
from scipy.stats import sem, t
import json

## 1. ******* CARGAMOS Y CONCATENAMOS LOS DATOS PARA EL/LA SUJETO EN CUESTIÓN *******
with open('codes\\parameters.json', 'r') as f:
    parameters = json.load(f)

sujeto = 8
sesion = 2
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
chizq_i = [eeg_concatenados.ch_names.index(chs_names[0])] #índice canal izquierdo
chder_i = [eeg_concatenados.ch_names.index(chs_names[1])] #índice canal derecho
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
# trials_iz_pretask = clase_izquierda.copy().crop(tmin=baseline_pretask[0], tmax=baseline_pretask[1])
trials_izq_task = clase_izquierda.copy().crop(tmin=baseline_task[0], tmax=baseline_task[1])
trials_izq_postask = clase_izquierda.copy().crop(tmin=baseline_postask[0], tmax=baseline_postask[1])
trials_der_rest = clase_derecha.copy().crop(tmin=baseline_rest[0], tmax=baseline_rest[1])
# trials_der_pretask = clase_derecha.copy().crop(tmin=baseline_pretask[0], tmax=baseline_pretask[1])
trials_der_task = clase_derecha.copy().crop(tmin=baseline_task[0], tmax=baseline_task[1])
trials_der_postask = clase_derecha.copy().crop(tmin=baseline_postask[0], tmax=baseline_postask[1])

izq_rest_multi_orig = mne.time_frequency.tfr_array_multitaper(trials_izq_rest.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
# izq_pretask_multi_orig = mne.time_frequency.tfr_array_multitaper(trials_iz_pretask.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
izq_task_multi_orig = mne.time_frequency.tfr_array_multitaper(trials_izq_task.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
izq_post_multi_orig = mne.time_frequency.tfr_array_multitaper(trials_izq_postask.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
der_rest_multi_orig = mne.time_frequency.tfr_array_multitaper(trials_der_rest.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
# der_pretask_multi_orig = mne.time_frequency.tfr_array_multitaper(trials_der_pretask.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
der_task_multi_orig = mne.time_frequency.tfr_array_multitaper(trials_der_task.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)
der_post_multi_orig = mne.time_frequency.tfr_array_multitaper(trials_der_postask.get_data(), freqs=freqs,sfreq=sfreq,n_cycles=n_cycles, time_bandwidth=2.0,output="power",n_jobs=1)

db = False
if db:
    izq_rest_multi = 20*np.log10(izq_rest_multi_orig)
    # izq_pretask_multi = 20*np.log10(izq_pretask_multi_orig)
    izq_task_multi = 20*np.log10(izq_task_multi_orig)
    izq_post_multi = 20*np.log10(izq_post_multi_orig)
    der_rest_multi = 20*np.log10(der_rest_multi_orig)
    # der_pretask_multi = 20*np.log10(der_pretask_multi_orig)
    der_task_multi = 20*np.log10(der_task_multi_orig)
    der_post_multi = 20*np.log10(der_post_multi_orig)
else:
    izq_rest_multi = izq_rest_multi_orig
    # izq_pretask_multi = izq_pretask_multi_orig
    izq_task_multi = izq_task_multi_orig
    izq_post_multi = izq_post_multi_orig
    der_rest_multi = der_rest_multi_orig
    # der_pretask_multi = der_pretask_multi_orig
    der_task_multi = der_task_multi_orig
    der_post_multi = der_post_multi_orig

## ************** REACTIVE BAND ********************

# no_cluster = False

# if no_cluster:
#     reactive_psd_izq_rest = izq_rest_multi[:,chder_i,:,:].mean((0,2)) #promedio primero sobre trials y luego sobre canales
#     # reactive_psd_izq_pretask = izq_pretask_multi[:,chder_i,:,:].mean((0,2))
#     reactive_psd_izq_task = izq_task_multi[:,chder_i,:,:].mean((0,2))
#     reactive_psd_izq_postask = izq_post_multi[:,chder_i,:,:].mean((0,2))

#     reactive_psd_der_rest = der_rest_multi[:,chizq_i,:,:].mean((0,2))
#     # reactive_psd_der_pretask = der_pretask_multi[:,chizq_i,:,:].mean((0,2))
#     reactive_psd_der_task = der_task_multi[:,chizq_i,:,:].mean((0,2))
#     reactive_psd_der_postask = der_post_multi[:,chizq_i,:,:].mean((0,2))

# else:

reactive_psd_izq_rest = izq_rest_multi[:,cluster_izq,:,:].mean((0,1,3)) #promedio sobre trials, luego canales y luego tiempo
# reactive_psd_izq_pretask = izq_pretask_multi[:,cluster_izq,:,:].mean((0,1,3))
reactive_psd_izq_task = izq_task_multi[:,cluster_izq,:,:].mean((0,1,3))
reactive_psd_izq_postask = izq_post_multi[:,cluster_izq,:,:].mean((0,1,3))

reactive_psd_der_rest = der_rest_multi[:,cluster_der,:,:].mean((0,1,3))
# reactive_psd_der_pretask = der_pretask_multi[:,cluster_der,:,:].mean((0,1,3))
reactive_psd_der_task = der_task_multi[:,cluster_der,:,:].mean((0,1,3))
reactive_psd_der_postask = der_post_multi[:,cluster_der,:,:].mean((0,1,3))

diff = izq_task_multi_orig[:,cluster_der,:,:] - izq_rest_multi_orig[:,cluster_der,:,:]
diff_avg = np.mean(diff, axis=(1, 3))  # shape: (18, 513)
n_trials = diff_avg.shape[0]
confidence = 0.95

mean_diff = np.mean(diff_avg, axis=0) 
sem_diff = sem(diff_avg, axis=0)       # SEM: error estándar de la media
# Intervalo t para muestras pequeñas
h = sem_diff * t.ppf((1 + confidence) / 2, df=n_trials - 1)
global_mean = np.mean(mean_diff)
global_ci_upper = global_mean + np.mean(h)
global_ci_lower = global_mean - np.mean(h)

confidence = 0.99
n = reactive_psd_izq_task.shape[0] #n es el mismo para todas las variables a usar
t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df = n - 1)

#cálculo de diferencias, media e intervalo de confianza para clase izquierda sobre electrodos de la derecha
diff_tr_izq = reactive_psd_izq_task - reactive_psd_izq_rest #diferencia de potencia entre task y rest clase izquierda
# diff_tr_izq = 20*np.log10(reactive_psd_izq_task/reactive_psd_izq_rest) #en dB
mean_diff_tr_izq = np.mean(diff_tr_izq)
std_tr_izq = np.std(diff_tr_izq, ddof=1) / np.sqrt(n) #error estándar de la media (debemos dividir por n^(1/2))
confinter_tr_izq = t_crit * std_tr_izq #intervalo de confianza para task - rest

diff_pr_izq = reactive_psd_izq_postask - reactive_psd_izq_rest #diferencia de potencia entre post_task y rest clase izquierda
mean_diff_pr_izq = np.mean(diff_pr_izq)
std_pr_izq = np.std(diff_pr_izq, ddof=1) / np.sqrt(n) #error estándar de la media (debemos dividir por n^(1/2))
confinter_pr_izq = t_crit * std_pr_izq #intervalo de confianza para post_task - rest

# diff_prr_izq = reactive_psd_izq_pretask - reactive_psd_izq_rest #diferencia de potencia entre post_task y pre_task clase izquierda
# mean_diff_prr_izq = np.mean(diff_prr_izq)
# std_prr_izq = np.std(diff_prr_izq, ddof=1) / np.sqrt(n) #error estándar de la media (debemos dividir por n^(1/2))
# confinter_prr_izq = t_crit * std_prr_izq #intervalo de confianza para post_task - pre_task

##repito para derecha
diff_tr_der = reactive_psd_der_task - reactive_psd_der_rest #diferencia de potencia entre task y rest clase derecha
mean_diff_tr_der = np.mean(diff_tr_der)
std_tr_der = np.std(diff_tr_der) / np.sqrt(n)
confinter_tr_der = t_crit * std_tr_der

diff_pr_der = reactive_psd_der_postask - reactive_psd_der_rest #diferencia de potencia entre posta_task y rest clase derecha
mean_diff_pr_der = np.mean(diff_pr_der)
std_pr_der = np.std(diff_pr_der) / np.sqrt(n)
confinter_pr_der = t_crit * std_pr_der

# diff_prr_der = reactive_psd_der_pretask - reactive_psd_der_rest #diferencia de potencia entre post_task y pre_task clase derecha
# mean_diff_prr_der = np.mean(diff_prr_der)
# std_prr_der = np.std(diff_prr_der) / np.sqrt(n)
# confinter_prr_der = t_crit * std_prr_der

##vamos a intentar replicar la gráfica derecha y arriba de 45.2 del artículo 
##https://neupsykey.com/eeg-event-related-desynchronization-erd-and-event-related-synchronization-ers/#R1-45

crest, ctask, cposttask = "#5dade2", "#45b27b", "#e74c3c"
cdiff = "#fc948f"
figsize=(8, 10)
title_fontsize=14
label_fs=14

fig, axes = plt.subplots(4, 1, figsize=figsize, constrained_layout=True)
axes[0].plot(freqs, diff_tr_izq, label=r"${\Delta}{{\mu}^2}$", color=cdiff,)
axes[0].axhline(mean_diff_tr_izq, color='grey')
axes[0].axhline(mean_diff_tr_izq + confinter_tr_izq, color='grey', linestyle='-.')
axes[0].axhline(mean_diff_tr_izq - confinter_tr_izq, color='grey', linestyle='-.')
axes[0].spines['top'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].legend(loc="lower right", fontsize=label_fs)
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[0].set_title(f"A1: {baseline_task} segundos", loc='left')

axes[1].plot(freqs, reactive_psd_izq_rest, label="R", linestyle=':', linewidth=1) 
axes[1].plot(freqs, reactive_psd_izq_task, label="A1", color=ctask, linewidth=2)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[1].legend(loc="upper right", fontsize=label_fs)

axes[2].plot(freqs, diff_pr_izq, label=r"${\Delta}{{\mu}^2}$", color=cdiff)
axes[2].axhline(mean_diff_pr_izq, color='grey')
axes[2].axhline(mean_diff_pr_izq + confinter_tr_izq, color='grey', linestyle='-.')
axes[2].axhline(mean_diff_pr_izq - confinter_tr_izq, color='grey', linestyle='-.')
axes[2].spines['top'].set_visible(False)
axes[2].spines['bottom'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].legend(loc="lower right", fontsize=label_fs)
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[2].set_title(f"A2: {baseline_postask} segundos", loc='left')

axes[3].plot(freqs, reactive_psd_izq_rest, label="R", linestyle=':', linewidth=1)
axes[3].plot(freqs, reactive_psd_izq_postask, label="A2", color=cposttask, linewidth=2)
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[3].legend(loc="upper right", fontsize=label_fs)

plt.suptitle(f"IZQUIERDA {tipo_sesion} - Suj. {sujeto}", fontsize=title_fontsize)
plt.show()


## Repito gráfico para clase derecha
fig, axes = plt.subplots(4, 1, figsize=figsize, constrained_layout=True)
axes[0].plot(freqs, diff_tr_der, label=r"${\Delta}{{\mu}^2}$", color=cdiff,)
axes[0].axhline(mean_diff_tr_der, color='grey')
axes[0].axhline(mean_diff_tr_der + confinter_tr_der, color='grey', linestyle='-.')
axes[0].axhline(mean_diff_tr_der - confinter_tr_der, color='grey', linestyle='-.')
axes[0].spines['top'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].legend(loc="lower right", fontsize=label_fs)
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[0].set_title(f"A1: {baseline_task} segundos", loc='left')

axes[1].plot(freqs, reactive_psd_der_rest, label="R", linestyle=':', linewidth=1) 
axes[1].plot(freqs, reactive_psd_der_task, label="A1", color=ctask, linewidth=2)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[1].legend(loc="upper right", fontsize=label_fs)

axes[2].plot(freqs, diff_pr_izq, label=r"${\Delta}{{\mu}^2}$", color=cdiff)
axes[2].axhline(mean_diff_pr_der, color='grey')
axes[2].axhline(mean_diff_pr_der + confinter_tr_der, color='grey', linestyle='-.')
axes[2].axhline(mean_diff_pr_der - confinter_tr_der, color='grey', linestyle='-.')
axes[2].spines['top'].set_visible(False)
axes[2].spines['bottom'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].legend(loc="lower right", fontsize=label_fs)
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[2].set_title(f"A2: {baseline_postask} segundos", loc='left')

axes[3].plot(freqs, reactive_psd_der_rest, label="R", linestyle=':', linewidth=1)
axes[3].plot(freqs, reactive_psd_der_postask, label="A2", color=cposttask, linewidth=2)
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[3].legend(loc="upper right", fontsize=label_fs)

plt.suptitle(f"DERECHA {tipo_sesion} - Suj. {sujeto}", fontsize=title_fontsize)
plt.show()