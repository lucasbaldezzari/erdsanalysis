import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs, getIntervalos
from scipy.stats import sem, t
import json
import os
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import fdrcorrection

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

clase_izquierda_cizq = epocas["IZQUIERDA"]#.pick_channels(parameters["cluster_electrodos_izq"])
clase_derecha_cizq = epocas["DERECHA"]#.pick_channels(parameters["cluster_electrodos_izq"])
clase_izquierda_cder = epocas["IZQUIERDA"]#.pick_channels(parameters["cluster_electrodos_der"])
clase_derecha_cder = epocas["DERECHA"]#.pick_channels(parameters["cluster_electrodos_der"])


## 3. ************************ OBTENGO EL TFRMULTIPAPER ************************
delta_freq = 0.2
freqs = np.arange(l_freq, h_freq, delta_freq)  # Frecuencias a filtrar (Hz)
n_cycles = freqs / 1.5
baseline = parameters["baseline_rest"]  # Intervalo de tiempo para el baseline

powerI_ci = clase_izquierda_cizq.compute_tfr(method="multitaper", freqs=freqs,
                                             n_cycles=n_cycles, time_bandwidth=2.0,
                                             n_jobs=1, average=False).apply_baseline(baseline=baseline, mode='percent')
powerD_ci = clase_derecha_cizq.compute_tfr(method="multitaper", freqs=freqs,
                                             n_cycles=n_cycles, time_bandwidth=2.0,
                                             n_jobs=1, average=False).apply_baseline(baseline=baseline, mode='percent')
powerI_cd = clase_izquierda_cder.compute_tfr(method="multitaper", freqs=freqs,
                                             n_cycles=n_cycles, time_bandwidth=2.0,
                                             n_jobs=1, average=False).apply_baseline(baseline=baseline, mode='percent')
powerD_cd = clase_derecha_cder.compute_tfr(method="multitaper", freqs=freqs,
                                             n_cycles=n_cycles, time_bandwidth=2.0,
                                             n_jobs=1, average=False).apply_baseline(baseline=baseline, mode='percent')

# ------- FUNCIÓN PARA r² ------
def compute_r2(x1, x2):
    mean_diff = x1.mean(axis=0) - x2.mean(axis=0)
    var_sum = x1.var(axis=0) + x2.var(axis=0)
    return (mean_diff**2) / var_sum

# ------- CALCULAR r²(f) por canal -------
ch_names = powerI_cd.info['ch_names']
n_channels = len(ch_names)
n_freqs = len(freqs)
r2 = np.zeros((n_channels, n_freqs))

band_avg = True
# Para cada canal y frecuencia
for ch_idx in range(n_channels):
    for f_idx in range(n_freqs):
        data_I = powerI_cd.data[:, ch_idx, f_idx, :]
        data_D = powerD_ci.data[:, ch_idx, f_idx, :]

        if band_avg:
            data_I = data_I[:, (powerI_cd.times >= tmin) & (powerI_cd.times <= tmax)].mean(axis=1)
            data_D = data_D[:, (powerD_cd.times >= tmin) & (powerD_cd.times <= tmax)].mean(axis=1)
        else:
            # Promediar en todo el tiempo
            data_I = data_I.mean(axis=1)
            data_D = data_D.mean(axis=1)

        r2[ch_idx, f_idx] = compute_r2(data_I, data_D)

layout = mne.find_layout(powerI_cd.info, ch_type='eeg')
for f_idx, f in enumerate(freqs[5:33:5]):
    r2_vals = r2[:, f_idx]
    mne.viz.plot_topomap(r2_vals, powerI_cd.info, cmap='coolwarm', contours=6,
                         show=False,)
    plt.title(f"r² @ {f:.1f} Hz")
    # plt.pause(0.3)  # para visualizar uno a uno

plt.show()

# ------- CALCULAR r²(f) por canal -------

# indexes = np.where((powerI_cd.freqs >= 8) & (powerI_cd.freqs <= 32))[0]
# layout = mne.find_layout(powerI_cd.info, ch_type='eeg')
# for f_idx, f in enumerate([[10]]):
#     r2_vals = r2[:,indexes].mean(1)#r2[:, f_idx]
#     mne.viz.plot_topomap(r2_vals, powerI_cd.info, cmap="Reds", contours=6,
#                          show=False,)
#     plt.title(f"r² @ {f:.1f} Hz")
#     # plt.pause(0.3)  # para visualizar uno a uno

# plt.show()