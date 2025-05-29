import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateSubjectsEEG, getHilbertERDS
from codes.utils import Baseline
import json
import os

## 1. ******* CARGAMOS Y CONCATENAMOS LOS DATOS PARA EL/LA SUJETO EN CUESTIÓN *******
##cargo codes\\parameters.json
with open('codes\\parameters.json', 'r') as f:
    parameters = json.load(f)

sujetos = [1,2,4,5,6,7,8,9] #parameters["sujeto"]
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
root_path = os.path.join("datasets", f"all_subjects","figures")
if not os.path.exists(root_path):
    os.makedirs(root_path)

data = concatenateSubjectsEEG(sujetos, sesion, apply_ica=True).drop_channels(channels_to_drop, "ignore")

for banda in ( parameters["banda_mu"], parameters["banda_beta"],parameters["banda_completa"]):
    eeg_concatenados = data.copy()
    # banda = parameters["banda_mu"]
    l_freq, h_freq = banda
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
    eventos = mne.pick_events(raw_eventos[0], include=[1,2])

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
    ws = 256
    erds_izq = getHilbertERDS(clase_izquierda, baseline, apply_smooth=True, window_smoothing=ws, mean_trials=True)
    erds_der = getHilbertERDS(clase_derecha, baseline, apply_smooth=True, window_smoothing=ws, mean_trials=True)


    c_ei, c_ed = parameters["colores_clases"] #colores para electrodos izquierdo y derecho
    cmap = parameters["cmap_topomaps"]
    sombra_fadein = "#fdf88c"
    times = clase_izquierda.times
    ti, tf = parameters["tiempo_plots"] #tiempo inicial y final para graficar
    idx = np.where((times >= ti) & (times <= tf))[0]
    i_times = np.where((times >= ti) & (times <= tf))[0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # axes[0].plot(times[idx], erds_izq[cluster_izq].mean(0)[idx], label="IZQUIERDA", color=c_ei, linewidth=2)
    # axes[0].fill_between(
    #     times[idx],
    #     erds_izq[cluster_izq].mean(0)[idx] - erds_izq[cluster_izq].std(0)[idx],
    #     erds_izq[cluster_izq].mean(0)[idx] + erds_izq[cluster_izq].std(0)[idx],
    #     color=c_ei,
    #     alpha=0.3,)
    axes[0].plot(times[idx], erds_der[cluster_izq].mean(0)[idx], label="DERECHA", color=c_ed, linewidth=2)
    axes[0].fill_between(
        times[idx],
        erds_der[cluster_izq].mean(0)[idx] - erds_der[cluster_izq].std(0)[idx],
        erds_der[cluster_izq].mean(0)[idx] + erds_der[cluster_izq].std(0)[idx],
        color=c_ed,
        alpha=0.3,)
    axes[0].axvline(0, color="k", linestyle="--", label="Cue onset")
    axes[0].axhline(0, color="grey", linestyle="--")
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel(r"Cambio voltaje (%)")
    axes[0].set_title(f"ERDS% Cluster Izq - ({l_freq}-{h_freq})Hz - {tipo_sesion}", fontsize=12)
    std_izq = erds_izq[cluster_izq].std(0)[idx]
    std_der = erds_der[cluster_izq].std(0)[idx]
    mean_izq = erds_izq[cluster_izq].mean(0)[idx]
    mean_der = erds_der[cluster_izq].mean(0)[idx]
    ymin = min(mean_izq.min() - std_izq.min(), mean_der.min() - std_der.min())
    ymax = max(mean_izq.max() + std_izq.max(), mean_der.max() + std_der.max())
    axes[0].fill_between(times, ymin, ymax, where=(times >= -0.25) & (times <= 0), color=sombra_fadein, alpha=0.2, label = "fade-in")
    axes[0].fill_between(times, ymin, ymax, where=(times >= 0) & (times<= 2), color="grey", alpha=0.2, label="Tarea")
    axes[0].legend(loc="lower right")

    axes[1].plot(times[idx], erds_izq[cluster_der].mean(0)[idx], label="IZQUIERDA", color=c_ei, linewidth=2)
    axes[1].fill_between(
        times[idx],
        erds_izq[cluster_der].mean(0)[idx] - erds_izq[cluster_der].std(0)[idx],
        erds_izq[cluster_der].mean(0)[idx] + erds_izq[cluster_der].std(0)[idx],
        color=c_ei,
        alpha=0.3,)
    # axes[1].plot(times[idx], erds_der[cluster_der].mean(0)[idx], label="DERECHA", color=c_ed, linewidth=2)
    # axes[1].fill_between(
    #     times[idx],
    #     erds_der[cluster_der].mean(0)[idx] - erds_der[cluster_der].std(0)[idx],
    #     erds_der[cluster_der].mean(0)[idx] + erds_der[cluster_der].std(0)[idx],
    #     color=c_ed,
    #     alpha=0.3,)
    axes[1].axvline(0, color="k", linestyle="--", label="Cue onset")
    axes[1].axhline(0, color="grey", linestyle="--")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel(r"Cambio voltaje (%)")
    axes[1].set_title(f"ERDS% Cluster DER - ({l_freq}-{h_freq})Hz - {tipo_sesion}", fontsize=12)
    std_izq = erds_izq[cluster_der].std(0)[idx]
    std_der = erds_der[cluster_der].std(0)[idx]
    mean_izq = erds_izq[cluster_der].mean(0)[idx]
    mean_der = erds_der[cluster_der].mean(0)[idx]
    ymin = min(mean_izq.min() - std_izq.min(), mean_der.min() - std_der.min())
    ymax = max(mean_izq.max() + std_izq.max(), mean_der.max() + std_der.max())
    axes[1].fill_between(times, ymin, ymax, where=(times >= -0.25) & (times <= 0), color=sombra_fadein, alpha=0.2, label = "fade-in")
    axes[1].fill_between(times, ymin, ymax, where=(times >= 0) & (times<= 2), color="gray", alpha=0.2, label="Tarea")
    axes[1].legend(loc="lower right")
    fig.canvas.manager.set_window_title(f"ERDS Grand Average - {tipo_sesion} - Banda {banda}.png")
    if save:
        plt.savefig(os.path.join(root_path, f"erds_voltaje_ga_{tipo_sesion}_{banda}.png"), dpi=350, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)