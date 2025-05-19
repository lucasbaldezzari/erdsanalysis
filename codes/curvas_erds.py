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

sujeto = parameters["sujeto"]
sesion = parameters["sesion"]
tipo_sesion = "Ejecutada" if sesion == 1 else "Imaginada"
sfreq = 512
channels_to_drop = parameters["channels_to_drop"]
pick = parameters["pick"]
confidence = parameters["confidence"]

eeg_concatenados = concatenateEEGs(sujeto, sesion, apply_ica=False).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")
# eeg_concatenados.plot_sensors(kind="topomap",show_names=True) ##probar con kind="3d"

# montage = {"C4": ["C2", "C6", "CP4", "FC4"],
#            "C3": ["C1", "C5", "CP3", "FC3"]
#            }

# lapfilter = LaplacianFilter(montage)
# lapfilter.apply(eeg_concatenados, inplace=True)

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
epocas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True,reject={"eeg":80})

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
clase_izquierda_avg = clase_izquierda.average()
clase_derecha_avg= clase_derecha.average()


## 3. ************************ CURVAS ERDS% USANDO HILBERT ************************
# Graficar curva ERDS para un canal específico (por ejemplo, C3)
  # Intervalo de tiempo para el baseline
baseline_rest = parameters["baseline_rest"]  # Intervalo de tiempo para el baseline
baseline_pretask = parameters["baseline_pretask"]  # Intervalo de tiempo para el baseline
baseline_task = parameters["baseline_task"]
baseline_postask = parameters["baseline_postask"]
baseline = Baseline(tuple(baseline_rest))
erds_izq = getHilbertERDS(clase_izquierda, baseline, apply_smooth=True, window_smoothing=102, mean_trials=True)
erds_der = getHilbertERDS(clase_derecha, baseline, apply_smooth=True, window_smoothing=102, mean_trials=True)


c_ei, c_ed = parameters["colores_clases"] #colores para electrodos izquierdo y derecho
cmap = parameters["cmap_topomaps"]
sombra_cue = "#fdf88c"
times = clase_izquierda.times
ti, tf = -1.6, tmax
idx = np.where((times >= ti) & (times <= tf))[0]
i_times = np.where((times >= ti) & (times <= tf))[0]
fig, axes = plt.subplots(1, 2, figsize=(5, 6))
axes[0].plot(times[idx], erds_izq[cluster_izq].mean(0)[idx], label="IZQUIERDA", color=c_ei, linewidth=2)
axes[0].plot(times[idx], erds_der[cluster_izq].mean(0)[idx], label="DERECHA", color=c_ed, linewidth=2)
axes[0].axvline(0, color="k", linestyle="--", label="Cue onset")
axes[0].axhline(0, color="grey", linestyle="--")
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_xlabel("Tiempo (s)")
axes[0].set_ylabel(r"ERDS (%) $\mu V$")
axes[0].set_title(f"EDRS% Cluster Izq - ({l_freq}-{h_freq})Hz - Sesión {tipo_sesion}", fontsize=14)
ymin = min(erds_izq[cluster_izq].mean(0)[idx].min(), erds_der[cluster_izq].mean(0)[idx].min())
ymax = max(erds_izq[cluster_izq].mean(0)[idx].max(), erds_der[cluster_izq].mean(0)[idx].max())
axes[0].fill_between(times, ymin, ymax, where=(times >= -0.5) & (times <= 0), color='grey', alpha=0.2)
axes[0].fill_between(times, ymin, ymax, where=(times >= 0) & (times<= 2), color=sombra_cue, alpha=0.2, label="Tarea")
axes[0].legend(loc="lower right")

axes[1].plot(times[idx], erds_izq[cluster_der].mean(0)[idx], label="IZQUIERDA", color=c_ei, linewidth=2)
axes[1].plot(times[idx], erds_der[cluster_der].mean(0)[idx], label="DERECHA", color=c_ed, linewidth=2)
axes[1].axvline(0, color="k", linestyle="--", label="Cue onset")
axes[1].axhline(0, color="grey", linestyle="--")
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].yaxis.tick_right()
axes[1].yaxis.set_label_position("right")
axes[1].set_xlabel("Tiempo (s)")
axes[1].set_ylabel(r"ERDS (%) $\mu V$")
axes[1].set_title(f"EDRS% Cluster DER - ({l_freq}-{h_freq})Hz - Sesión {tipo_sesion}", fontsize=14)
ymin = min(erds_izq[cluster_der].mean(0)[idx].min(), erds_der[cluster_der].mean(0)[idx].min())
ymax = max(erds_izq[cluster_der].mean(0)[idx].max(), erds_der[cluster_der].mean(0)[idx].max())
axes[1].fill_between(times, ymin, ymax, where=(times >= -0.5) & (times <= 0), color='grey', alpha=0.2)
axes[1].fill_between(times, ymin, ymax, where=(times >= 0) & (times<= 2), color=sombra_cue, alpha=0.2, label="Tarea")
axes[1].legend(loc="lower right")

plt.show()