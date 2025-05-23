import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs, getIntervalos
from scipy.stats import sem, t
import json
import os

## 1. ******* CARGAMOS Y CONCATENAMOS LOS DATOS PARA EL/LA SUJETO EN CUESTIÓN *******
##cargo codes\\parameters.json
with open('codes\\parameters.json', 'r') as f:
    parameters = json.load(f)

sujeto = parameters["sujeto"]
sesion = parameters["sesion"]
tipo_sesion = "Ejecutada" if sesion == 1 else "Imaginada"
sfreq = 512
channels_to_drop = parameters["channels_to_drop"]
pick = parameters["pick"]
confidence = parameters["confidence"]

##para mostrar y guardar gráficos
show = parameters["show_figures"]
save = parameters["save_figures"]

## folder a donde guardar los gráficos
root_path = os.path.join("datasets", f"sujeto_{sujeto}","figures")
if not os.path.exists(root_path):
    os.makedirs(root_path)

eeg_concatenados = concatenateEEGs(sujeto, sesion, runs=[1,2]).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")

l_freq, h_freq = parameters["banda_completa"]
eeg_concatenados.filter(l_freq=l_freq, h_freq=h_freq,
           picks="eeg", 
           method="fir", 
           phase="zero-double", 
           fir_window="hamming",
           filter_length="auto")


## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = parameters["duracion_trial"]
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
delta_freq = 0.1
freqs = np.arange(l_freq, h_freq, delta_freq)  # Frecuencias a filtrar (Hz)
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

## ------- Obtengo los PSD de cada segmento de interés  -------
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

reactive_psd_izq_rest = izq_rest_multi[:,cluster_izq,:,:].mean((0,1,3)) #promedio sobre trials, luego canales y luego tiempo
# reactive_psd_izq_pretask = izq_pretask_multi[:,cluster_izq,:,:].mean((0,1,3))
reactive_psd_izq_task = izq_task_multi[:,cluster_izq,:,:].mean((0,1,3))
reactive_psd_izq_postask = izq_post_multi[:,cluster_izq,:,:].mean((0,1,3))

reactive_psd_der_rest = der_rest_multi[:,cluster_der,:,:].mean((0,1,3))
# reactive_psd_der_pretask = der_pretask_multi[:,cluster_der,:,:].mean((0,1,3))
reactive_psd_der_task = der_task_multi[:,cluster_der,:,:].mean((0,1,3))
reactive_psd_der_postask = der_post_multi[:,cluster_der,:,:].mean((0,1,3))

### IZQUIERDA ###
##diferencia entre la tarea y rest
n_izq_trials = epocas["IZQUIERDA"].events.shape[0]

diff_tr_izq_avg = (izq_task_multi[:,cluster_der,:,:] - izq_rest_multi[:,cluster_der,:,:]).mean((1,3))
mean_diff_tr_izq = diff_tr_izq_avg.mean(0)
global_mean_diff_tr_izq = diff_tr_izq_avg.mean()
sem_diff_tr_izq_avg = sem(diff_tr_izq_avg, axis=0)
confinter_tr_izq = sem_diff_tr_izq_avg * t.ppf((1 + confidence) / 2, df=n_izq_trials - 1)

##diferencia entre tarea y postask izquierda
diff_tr_izq_postask_avg = (izq_post_multi[:,cluster_der,:,:] - izq_rest_multi[:,cluster_der,:,:]).mean((1,3))
mean_diff_pr_izq = diff_tr_izq_postask_avg.mean(0)
global_mean_diff_pr_izq = diff_tr_izq_postask_avg.mean()
sem_diff_pr_izq_avg = sem(diff_tr_izq_postask_avg, axis=0)
confinter_pr_izq = sem_diff_pr_izq_avg * t.ppf((1 + confidence) / 2, df=n_izq_trials - 1)

### DERECHA ###
##diferencia entre la tarea y rest derecha
n_der_trials = epocas["DERECHA"].events.shape[0]

diff_tr_der_avg = (der_task_multi[:,cluster_izq,:,:] - der_rest_multi[:,cluster_izq,:,:]).mean((1,3))
mean_diff_tr_der = diff_tr_der_avg.mean(0)
global_mean_diff_tr_der = diff_tr_der_avg.mean()
sem_diff_tr_der_avg = sem(diff_tr_der_avg, axis=0)
confinter_tr_der = sem_diff_tr_der_avg * t.ppf((1 + confidence) / 2, df=n_der_trials - 1)

##diferencia entre tarea y postask derecha
diff_tr_der_postask_avg = (der_post_multi[:,cluster_izq,:,:] - der_rest_multi[:,cluster_izq,:,:]).mean((1,3))
mean_diff_pr_der = diff_tr_der_postask_avg.mean(0)
global_mean_diff_pr_der = diff_tr_der_postask_avg.mean()
sem_diff_pr_der_avg = sem(diff_tr_der_postask_avg, axis=0)
confinter_pr_der = sem_diff_pr_der_avg * t.ppf((1 + confidence) / 2, df=n_der_trials - 1)

##vamos a intentar replicar la gráfica 45.2 del lado izquierdo y arriba de del artículo 
##https://neupsykey.com/eeg-event-related-desynchronization-erd-and-event-related-synchronization-ers/#R1-45

crest, ctask, cposttask = "#000000", "#f7525f", "#006b3c"
cdiff = "#5b0672"
figsize=(8, 10)
title_fontsize=14
label_fs=14

freqs_inter_izq = {}

# sig_freqs = list(freqsabove) + list(freqsbelow)
## ***** Gráfica para clase IZQUIERDA *****
fig, axes = plt.subplots(4, 1, figsize=figsize, constrained_layout=True)
axes[0].plot(freqs, mean_diff_tr_izq, label=r"${\Delta}{{\mu}^2}$", color=cdiff, linestyle='--')
axes[0].axhline(global_mean_diff_tr_izq, color='grey')
axes[0].axhline(global_mean_diff_tr_izq + confinter_tr_izq.mean(), color='grey', linestyle='-.')
axes[0].axhline(global_mean_diff_tr_izq - confinter_tr_izq.mean(), color='grey', linestyle='-.')
freqsabove = freqs[np.where(mean_diff_tr_izq > global_mean_diff_tr_izq + confinter_tr_izq.mean())[0]]
intervalos_above = getIntervalos(freqsabove.copy(), delta_freq)
freqs_inter_izq["izq_tr_abov"] = intervalos_above
for intervalo in intervalos_above:
    axes[0].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
freqsbelow = freqs[np.where(mean_diff_tr_izq < global_mean_diff_tr_izq - confinter_tr_izq.mean())[0]]
intervalos_below = getIntervalos(freqsbelow, delta_freq)
freqs_inter_izq["izq_tr_below"] = intervalos_below
for intervalo in intervalos_below:
    axes[0].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
axes[0].spines['top'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].legend(loc="lower right", fontsize=label_fs)
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[0].set_title(f"A1: {baseline_task} segundos", loc='left')

axes[1].plot(freqs, reactive_psd_izq_rest, label="R", linestyle=':', color = crest, linewidth=2) 
axes[1].plot(freqs, reactive_psd_izq_task, label="A1", color=ctask, linewidth=2)
for intervalo in intervalos_above:
    axes[1].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
    centro = np.mean(intervalo)
    freqi = intervalo[0]
    freqf = intervalo[-1]
    axes[1].text(centro, 0.5, f"{freqi}-{freqf} Hz", ha='center', va='bottom', fontsize=11, color='k', rotation=45)
for intervalo in intervalos_below:
    axes[1].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
    centro = np.mean(intervalo)
    freqi = intervalo[0]
    freqf = intervalo[-1]
    axes[1].text(centro, 0.5, f"{freqi}-{freqf} Hz", ha='center', va='bottom', fontsize=11, color='k', rotation=45)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[1].legend(loc="upper right", fontsize=label_fs)

axes[2].plot(freqs, mean_diff_pr_izq, label=r"${\Delta}{{\mu}^2}$", color=cdiff, linestyle='--')
axes[2].axhline(global_mean_diff_pr_izq, color='grey')
axes[2].axhline(global_mean_diff_pr_izq + confinter_pr_izq.mean(), color='grey', linestyle='-.')
axes[2].axhline(global_mean_diff_pr_izq - confinter_pr_izq.mean(), color='grey', linestyle='-.')
freqsabove = freqs[np.where(mean_diff_pr_izq > global_mean_diff_pr_izq + confinter_pr_izq.mean())[0]]
intervalos_above = getIntervalos(freqsabove, delta_freq)
freqs_inter_izq["izq_pr_abov"] = intervalos_above
for intervalo in intervalos_above:
    axes[2].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
freqsbelow = freqs[np.where(mean_diff_pr_izq < global_mean_diff_pr_izq - confinter_pr_izq.mean())[0]]
intervalos_below = getIntervalos(freqsbelow, delta_freq)
freqs_inter_izq["izq_pr_below"] = intervalos_below
for intervalo in intervalos_below:
    axes[2].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
axes[2].spines['top'].set_visible(False)
axes[2].spines['bottom'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].legend(loc="lower right", fontsize=label_fs)
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[2].set_title(f"A2: {baseline_postask} segundos", loc='left')

axes[3].plot(freqs, reactive_psd_izq_rest, label="R", linestyle=':', color = crest, linewidth=2)
axes[3].plot(freqs, reactive_psd_izq_postask, label="A2", color=cposttask, linewidth=2)
for intervalo in intervalos_above:
    axes[3].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
    centro = np.mean(intervalo)
    freqi = intervalo[0]
    freqf = intervalo[-1]
    axes[3].text(centro, 0.5, f"{freqi}-{freqf} Hz", ha='center', va='bottom', fontsize=11, color='k', rotation=45)
for intervalo in intervalos_below:
    axes[3].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
    centro = np.mean(intervalo)
    freqi = intervalo[0]
    freqf = intervalo[-1]
    axes[3].text(centro, 0.5, f"{freqi}-{freqf} Hz", ha='center', va='bottom', fontsize=11, color='k', rotation=45)
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[3].legend(loc="upper right", fontsize=label_fs)
axes[3].set_xlabel("Frecuencia (Hz)", fontsize=label_fs)
fig.canvas.manager.set_window_title(f"IZQUIERDA {tipo_sesion} - Suj. {sujeto}")
plt.suptitle(f"IZQUIERDA {tipo_sesion} - Suj. {sujeto}", fontsize=title_fontsize)
if save:
    plt.savefig(os.path.join(root_path, f"spectral_izq_s{sujeto}_{tipo_sesion}.png"), dpi=350)
if show:
    plt.show()
plt.close(fig)

## ***** Gráfica para clase DERECHA *****
freqs_inter_der = {}
fig, axes = plt.subplots(4, 1, figsize=figsize, constrained_layout=True)
axes[0].plot(freqs, mean_diff_tr_der, label=r"${\Delta}{{\mu}^2}$", color=cdiff, linestyle='--')
axes[0].axhline(global_mean_diff_tr_der, color='grey')
axes[0].axhline(global_mean_diff_tr_der + confinter_tr_der.mean(), color='grey', linestyle='-.')
axes[0].axhline(global_mean_diff_tr_der - confinter_tr_der.mean(), color='grey', linestyle='-.')
freqsabove = freqs[np.where(mean_diff_tr_der > global_mean_diff_tr_der + confinter_tr_der.mean())[0]]
intervalos_above = getIntervalos(freqsabove.copy(), delta_freq)
freqs_inter_der["der_tr_abov"] = intervalos_above
for intervalo in intervalos_above:
    axes[0].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
freqsbelow = freqs[np.where(mean_diff_tr_der < global_mean_diff_tr_der - confinter_tr_der.mean())[0]]
intervalos_below = getIntervalos(freqsbelow, delta_freq)
freqs_inter_der["der_tr_below"] = intervalos_below
for intervalo in intervalos_below:
    axes[0].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
axes[0].spines['top'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].legend(loc="lower right", fontsize=label_fs)
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[0].set_title(f"A1: {baseline_task} segundos", loc='left')

axes[1].plot(freqs, reactive_psd_der_rest, label="R", linestyle=':', color = crest, linewidth=2) 
axes[1].plot(freqs, reactive_psd_der_task, label="A1", color=ctask, linewidth=2)
for intervalo in intervalos_above:
    axes[1].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
    centro = np.mean(intervalo)
    freqi = intervalo[0]
    freqf = intervalo[-1]
    axes[1].text(centro, 0.5, f"{freqi}-{freqf} Hz", ha='center', va='bottom', fontsize=11, color='k', rotation=45)
for intervalo in intervalos_below:
    axes[1].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
    centro = np.mean(intervalo)
    freqi = intervalo[0]
    freqf = intervalo[-1]
    axes[1].text(centro, 0.5, f"{freqi}-{freqf} Hz", ha='center', va='bottom', fontsize=11, color='k', rotation=45)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[1].legend(loc="upper right", fontsize=label_fs)

axes[2].plot(freqs, mean_diff_pr_der, label=r"${\Delta}{{\mu}^2}$", color=cdiff, linestyle='--')
axes[2].axhline(global_mean_diff_pr_der, color='grey')
axes[2].axhline(global_mean_diff_pr_der + confinter_pr_der.mean(), color='grey', linestyle='-.')
axes[2].axhline(global_mean_diff_pr_der - confinter_pr_der.mean(), color='grey', linestyle='-.')
freqsabove = freqs[np.where(mean_diff_pr_der > global_mean_diff_pr_der + confinter_pr_der.mean())[0]]
intervalos_above = getIntervalos(freqsabove, delta_freq)
freqs_inter_der["der_pr_abov"] = intervalos_above
for intervalo in intervalos_above:
    axes[2].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
freqsbelow = freqs[np.where(mean_diff_pr_der < global_mean_diff_pr_der - confinter_pr_der.mean())[0]]
intervalos_below = getIntervalos(freqsbelow, delta_freq)
freqs_inter_der["der_pr_below"] = intervalos_below
for intervalo in intervalos_below:
    axes[2].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
axes[2].spines['top'].set_visible(False)
axes[2].spines['bottom'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].legend(loc="lower right", fontsize=label_fs)
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axes[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[2].set_title(f"A2: {baseline_postask} segundos", loc='left')

axes[3].plot(freqs, reactive_psd_der_rest, label="R", linestyle=':', color = crest, linewidth=2)
axes[3].plot(freqs, reactive_psd_der_postask, label="A2", color=cposttask, linewidth=2)
for intervalo in intervalos_above:
    axes[3].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
    centro = np.mean(intervalo)
    freqi = intervalo[0]
    freqf = intervalo[-1]
    axes[3].text(centro, 0.5, f"{freqi}-{freqf} Hz", ha='center', va='bottom', fontsize=11, color='k', rotation=45)
for intervalo in intervalos_below:
    axes[3].axvspan(intervalo[0], intervalo[-1], color='grey', alpha=0.2)
    centro = np.mean(intervalo)
    freqi = intervalo[0]
    freqf = intervalo[-1]
    axes[3].text(centro, 0.5, f"{freqi}-{freqf} Hz", ha='center', va='bottom', fontsize=11, color='k', rotation=45)
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
axes[3].legend(loc="upper right", fontsize=label_fs)
axes[3].set_xlabel("Frecuencia (Hz)", fontsize=label_fs)
plt.suptitle(f"DERECHA {tipo_sesion} - Suj. {sujeto}", fontsize=title_fontsize)
fig.canvas.manager.set_window_title(f"DERECHA {tipo_sesion} - Suj. {sujeto}")
if save:
    plt.savefig(os.path.join(root_path, f"spectral_der_s{sujeto}_{tipo_sesion}.png"), dpi=350)
if show:
    plt.show()
plt.close(fig)
