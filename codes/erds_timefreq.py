import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs, getHilbertERDS
from mne.time_frequency import tfr_morlet
from codes.utils import Baseline
from matplotlib.colors import TwoSlopeNorm
from codes.utils import LaplacianFilter, EnvolventeEEG
import json

## 1. ******* CARGAMOS Y CONCATENAMOS LOS DATOS PARA EL/LA SUJETO EN CUESTIÓN *******
##cargo codes\\parameters.json
with open('codes\\parameters.json', 'r') as f:
    parameters = json.load(f)
sujeto = 8
sesion = 2
sfreq = 512

tipo_sesion = "Ejecutada" if sesion == 1 else "Imaginada"

channels_to_drop = ["FP1","FP2","FPz","Fz","F8","F7","AF3","AF4","AF5","AF7","AF8","T7","T8","F9","F10"]
pick = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6",]

eeg_concatenados = concatenateEEGs(sujeto, sesion, apply_ica=False).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")
# eeg_concatenados.plot_sensors(kind="topomap",show_names=True) ##probar con kind="3d"

# plotEEG(eeg_concatenados, show=True, scalings=40, bad_color = "red")

l_freq, h_freq = parameters["banda_completa"]
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
chizq_i = eeg_concatenados.ch_names.index(chs_names[0]) #índice canal izquierdo
chder_i = eeg_concatenados.ch_names.index(chs_names[1]) #índice canal derecho
clase_izquierda = epocas["IZQUIERDA"]
clase_derecha = epocas["DERECHA"]
clase_izquierda_avg = clase_izquierda.average()
clase_derecha_avg= clase_derecha.average()


## 3. ************************ ANALISIS TIEMPO-FRECUENCIA ************************
### Aplicar el análisis de Morlet para obtener la potencia en el rango de frecuencias deseado y luego tomar los datos, aplicar el baseline usando el 
### los datos del tiempo baseline y así rasignar la data al objeto MNE.

freqs = np.arange(l_freq, h_freq, 0.2)  # Frecuencias a filtrar (Hz)

tfr_izq = tfr_morlet(clase_izquierda, freqs=freqs, n_cycles=freqs/2., return_itc=False, average=False)
tfr_der = tfr_morlet(clase_derecha, freqs=freqs, n_cycles=freqs/2., return_itc=False, average=False)

baseline_rest = parameters["baseline_rest"]  # Intervalo de tiempo para el baseline
baseline_pretask = parameters["baseline_pretask"]  # Intervalo de tiempo para el baseline
baseline_task = parameters["baseline_task"]
baseline_postask = parameters["baseline_postask"]

tfr_izq_base = tfr_izq.copy().apply_baseline(baseline_rest,mode="percent")
tfr_der_base = tfr_der.copy().apply_baseline(baseline_rest,mode="percent")

baseline = Baseline(tuple(baseline_rest))

tfr_der_avg = tfr_der.average().apply_baseline(baseline_rest,mode="percent")
tfr_izq_avg = tfr_izq.average().apply_baseline(baseline_rest,mode="percent")

# tfr_der_avg.plot(picks=chs_names)
# tfr_der_avg.plot_topomap(tmin=0, tmax=1,fmin=8,fmax=32,cmap="RdBu_r")

banda = parameters["banda_mu"]  # Banda de frecuencias a filtrar (Hz)
#indices donde freqs sea mayor a 10 y menor a 12
i_freqs = np.where((freqs >= banda[0]) & (freqs <= banda[1]))[0]

cluster_izq = [eeg_concatenados.ch_names.index(ch) for ch in parameters["cluster_electrodos_izq"]]
cluster_der = [eeg_concatenados.ch_names.index(ch) for ch in parameters["cluster_electrodos_der"]]

no_cluster = True

if no_cluster:
    tfr_i_filt_c3 = tfr_izq_base.data[:,:,i_freqs,:].mean(axis=2)[:,chizq_i,:].mean(axis=0)
    tfr_i_filt_c4 = tfr_izq_base.data[:,:,i_freqs,:].mean(axis=2)[:,chder_i,:].mean(axis=0)
    tfr_d_filt_c3 = tfr_der_base.data[:,:,i_freqs,:].mean(axis=2)[:,chizq_i,:].mean(axis=0)
    tfr_d_filt_c4 = tfr_der_base.data[:,:,i_freqs,:].mean(axis=2)[:,chder_i,:].mean(axis=0)
    title_izq = chizq_n
    title_der = chder_n
else:
    tfr_i_filt_c3 = tfr_izq_base.data[:,:,i_freqs,:].mean(axis=2)[:,cluster_izq,:].mean(axis=1).mean(axis=0)
    tfr_i_filt_c4 = tfr_izq_base.data[:,:,i_freqs,:].mean(axis=2)[:,cluster_der,:].mean(axis=1).mean(axis=0)
    tfr_d_filt_c3 = tfr_der_base.data[:,:,i_freqs,:].mean(axis=2)[:,cluster_izq,:].mean(axis=1).mean(axis=0)
    tfr_d_filt_c4 = tfr_der_base.data[:,:,i_freqs,:].mean(axis=2)[:,cluster_der,:].mean(axis=1).mean(axis=0)
    title_izq = parameters["cluster_electrodos_izq"]
    title_der = parameters["cluster_electrodos_der"]

window_size = 512
tfr_d_filt_c4_smooth = np.convolve(tfr_d_filt_c4, np.ones(window_size)/window_size, mode='same')
tfr_d_filt_c3_smooth = np.convolve(tfr_d_filt_c3, np.ones(window_size)/window_size, mode='same')
tfr_i_filt_c4_smooth = np.convolve(tfr_i_filt_c4, np.ones(window_size)/window_size, mode='same')
tfr_i_filt_c3_smooth = np.convolve(tfr_i_filt_c3, np.ones(window_size)/window_size, mode='same')

c_ei, c_ed = parameters["colores_clases"] #colores para electrodos izquierdo y derecho
cmap = parameters["cmap_topomaps"]
fmin, fmax = banda
ti, tf = -1.6, tmax
times = tfr_izq.times
i_times = np.where((times >= ti) & (times <= tf))[0]
fig, axes = plt.subplots(1, 4, figsize=(17, 6))
axes[0].plot(times[i_times], tfr_i_filt_c3_smooth[i_times], label="IZQ", color=c_ei, linewidth=2)
axes[0].plot(times[i_times], tfr_d_filt_c3_smooth[i_times], label="DER", color=c_ed, linewidth=2)
axes[0].set_xlabel('Tiempo (s)')
axes[0].set_ylabel('Potencia (%)')
#agrego una sombra entre los tiempos -0.5 y 0
ymin = min(tfr_d_filt_c3_smooth[i_times].min(), tfr_i_filt_c3_smooth[i_times].min())
ymax = max(tfr_d_filt_c3_smooth[i_times].max(), tfr_i_filt_c3_smooth[i_times].max())
offset = 0
axes[0].fill_between(times, ymin-offset, ymax+offset, where=(times >= -0.5) & (times <= 0), color='grey', alpha=0.2)
axes[0].fill_between(times, ymin-offset, ymax+offset, where=(times >= 0) & (times<= 2), color='#725ba0', alpha=0.2)
axes[0].axvline(0, color='k', linestyle='-', label='Cue')
axes[0].axvline(-0.5, color='grey', linestyle='--')
axes[0].axvline(2, color='grey', linestyle='--')
axes[0].axhline(0, color='grey', linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].legend(loc="upper right") 
axes[0].set_title(title_izq)

axes[3].plot(times[i_times], tfr_i_filt_c4_smooth[i_times], label="IZQ", color=c_ei, linewidth=2)
axes[3].plot(times[i_times], tfr_d_filt_c4_smooth[i_times], label="DER", color=c_ed, linewidth=2)
axes[3].set_xlabel('Tiempo (s)')
axes[3].set_ylabel('Potencia (%)')
# axes[3].set_ylabel("Potencia (%)", labelpad=10) 
#agrego una sombra entre los tiempos -0.5 y 0
axes[3].yaxis.set_label_position("right")
ymin = min(tfr_d_filt_c4_smooth[i_times].min(), tfr_i_filt_c4_smooth[i_times].min())
ymax = max(tfr_d_filt_c4_smooth[i_times].max(), tfr_i_filt_c4_smooth[i_times].max())
axes[3].fill_between(times, ymin-offset, ymax+offset, where=(times >= -0.5) & (times <= 0), color='grey', alpha=0.2)
axes[3].fill_between(times, ymin-offset, ymax+offset, where=(times >= 0) & (times <= 2), color='#725ba0', alpha=0.2)
axes[3].axvline(0, color='k', linestyle='-', label='Cue')
axes[3].axvline(-0.5, color='grey', linestyle='--')
axes[3].axvline(2, color='grey', linestyle='--')
axes[3].axhline(0, color='grey', linestyle='--')
axes[3].spines['top'].set_visible(False)
axes[3].spines['left'].set_visible(False)
axes[3].yaxis.tick_right()
axes[3].legend(loc="upper right")
axes[3].set_title(title_der)

tfr_izq.average().apply_baseline(baseline_rest,mode="percent").plot_topomap(tmin=0, tmax=1,fmin=fmin,fmax=fmax,colorbar=False,
                                                                            cmap=cmap,show=False,axes=axes[1],contours=8,cbar_fmt="%.2f")
tfr_der.average().apply_baseline(baseline_rest,mode="percent").plot_topomap(tmin=0, tmax=1,fmin=fmin,fmax=fmax,colorbar=False,
                                                                            cmap=cmap,show=False,axes=axes[2],contours=8)
axes[1].set_title("IZQUIERDA")
axes[2].set_title("DERECHA")
# plt.suptitle(f"${tipo_sesion}$ - {banda[0]}-{banda[1]}Hz",fontsize=18)
plt.tight_layout()
plt.show()