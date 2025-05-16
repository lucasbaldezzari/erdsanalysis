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

electrodos = ["CP3","CP4"]
e1_index = eeg_concatenados.ch_names.index(electrodos[0])
e2_index = eeg_concatenados.ch_names.index(electrodos[1])
clase_izquierda = epocas["IZQUIERDA"]
clase_derecha = epocas["DERECHA"]
clase_izquierda_avg = clase_izquierda.average()
clase_derecha_avg= clase_derecha.average()


## 3. ************************ ANALISIS TIEMPO-FRECUENCIA ************************
### Aplicar el análisis de Morlet para obtener la potencia en el rango de frecuencias deseado y luego tomar los datos, aplicar el baseline usando el 
### los datos del tiempo baseline y así rasignar la data al objeto MNE.

baseline = Baseline((-1.5, -0.5))

freqs = np.arange(l_freq, h_freq, 0.5)  # Frecuencias a filtrar (Hz)
baseline_rest = (-2, -1)  # Intervalo de tiempo para el baseline
baseline_pretask = (-0.5, 0)  # Intervalo de tiempo para el baseline
baseline_task = (0, 1)
baseline_postask = (2, 3)

tfr_izq = tfr_morlet(clase_izquierda, freqs=freqs, n_cycles=freqs/2., return_itc=False, average=False)
tfr_der = tfr_morlet(clase_derecha, freqs=freqs, n_cycles=freqs/2., return_itc=False, average=False)

tfr_izq.average().apply_baseline(baseline_pretask,mode="percent").plot(picks=electrodos)

#indices donde freqs sea mayor a 10 y menor a 12
indices = np.where((freqs >= 8) & (freqs <= 12))[0]
electrodos = ["CP3","CP4"]
e1_index = eeg_concatenados.ch_names.index(electrodos[0])
e2_index = eeg_concatenados.ch_names.index(electrodos[1])

tfr_izq_data_1012 = tfr_izq.data[:, :, indices, :].mean(axis=2) ##media en el eje de frecuencias
##aplico baseline para cada trial
for trial in range(tfr_izq_data_1012.shape[0]):
    baseline_mean = baseline.apply(tfr_izq_data_1012[trial, :], tfr_izq.times)
    tfr_izq_data_1012[trial, :] = 100*(tfr_izq_data_1012[trial, :] - baseline_mean) / baseline_mean
tfr_izq_data_1012_c4 = tfr_izq_data_1012.mean(axis=0)[e2_index, :]
##repito para derecha
tfr_der_data_1012 = tfr_der.data[:, :, indices, :].mean(axis=2) ##media en el eje de frecuencias
for trial in range(tfr_der_data_1012.shape[0]):
    baseline_mean = baseline.apply(tfr_der_data_1012[trial, :], tfr_der.times)
    tfr_der_data_1012[trial, :] = 100*(tfr_der_data_1012[trial, :] - baseline_mean) / baseline_mean
tfr_der_data_1012_c3 = tfr_der_data_1012.mean(axis=0)[e1_index, :]

window_size = 512
color_izq, color_der = "#5dade2", "#e74c3c"
plt.figure(figsize=(10, 5))
tfr_izq_data_1012_c4 = np.convolve(tfr_izq_data_1012_c4, np.ones(window_size)/window_size, mode='same')
tfr_der_data_1012_c3 = np.convolve(tfr_der_data_1012_c3, np.ones(window_size)/window_size, mode='same')
plt.plot(tfr_izq.times, tfr_izq_data_1012_c4, label="IZQUIERDA - C4", color=color_izq, linewidth=2)
plt.plot(tfr_der.times, tfr_der_data_1012_c3, label="DERECHA - C3", color=color_der, linewidth=2)
#linea vertical en 0, en -0.5 y en 2
plt.axvline(0, color='k', linestyle='--', label='Cue onset')
plt.axvline(-0.5, color='grey', linestyle='--')
plt.axvline(2, color='grey', linestyle='--')
plt.axhline(0, color='grey', linestyle='--')
plt.xlabel('Tiempo (s)')
plt.ylabel('ERDS (%)')
plt.title('Curvas ERDS% en C3 y C4')
plt.grid()
plt.legend()
plt.show()