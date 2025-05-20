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

clase_izquierda_cizq = epocas["IZQUIERDA"].pick_channels(parameters["cluster_electrodos_izq"])
clase_derecha_cizq = epocas["DERECHA"].pick_channels(parameters["cluster_electrodos_izq"])
clase_izquierda_cder = epocas["IZQUIERDA"].pick_channels(parameters["cluster_electrodos_der"])
clase_derecha_cder = epocas["DERECHA"].pick_channels(parameters["cluster_electrodos_der"])


## 3. ************************ OBTENGO EL TFRMULTIPAPER ************************
delta_freq = 0.5
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


# ----- Parámetros de análisis -----
freq_bands = {"mu": parameters["banda_mu"],
              "beta": parameters["banda_beta"]}
ti, tf = 0,2 #duración del cue

# ----- Agrupo por banda y comparo -----
for band_name, (fmin, fmax) in freq_bands.items():
    band_inds = np.where((powerI_cd.freqs >= fmin) & (powerI_cd.freqs <= fmax))[0]
    times = (powerI_cd.times >= tmin) & (powerI_cd.times <= tmax)
    print(f"Banda: {band_name} ({fmin}-{fmax} Hz)")
    p_vals = []
    t_vals = []

    for ch_idx, ch in enumerate(parameters["cluster_electrodos_izq"]):
        # Promedio por trial en tiempo y frecuencia
        data_I = powerI_cd.data[:, ch_idx, band_inds][:, :, times].mean(axis=(1, 2))
        data_D = powerD_cd.data[:, ch_idx, band_inds][:, :, times].mean(axis=(1, 2))

        # Test t pareado
        t_stat, p_val = ttest_ind(data_I, data_D)
        t_vals.append(t_stat)
        p_vals.append(p_val)

        print(f"Canal {ch}: t={t_stat:.2f}, p={p_val:.4f}")

    # Corrección FDR
    reject, p_corrected = fdrcorrection(p_vals, alpha=0.05)
    for ch, pval, pcorr, rej in zip(parameters["cluster_electrodos_izq"], p_vals, p_corrected, reject):
        print(f"Canal {ch}: p_corr={pcorr:.4f}, {'significativo' if rej else 'ns'}")

