"""
Script para analizar la presencia de patrones ERDS en los datos de EEG.

Este script utiliza la biblioteca MNE-Python para cargar datos de EEG, aplicar un filtro,
calcular la potencia de la señal y graficar los resultados.
El análisis de ERDS (Event-Related Desynchronization) se utiliza para estudiar
la actividad cerebral relacionada con eventos específicos, como movimientos 
o estímulos visuales.

Se seguirán algunos de los pasos mostrados en el siguiente ejemplo de MNE-Python:
https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html
"""

## IMPORTAMOS LIBRERÍAS
from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
import h5py
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import read_ica
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test as pcluster_test

###PRIMEROS PASOS
sfreq = 512 # Frecuencia de muestreo
##cargamos los nombres de los electrodos del g.HIAMP
montage_df = pd.read_csv("codes\\ghiamp_montage.sfp",sep="\t",header=None)
ch_names = list(montage_df[0])
channels_to_remove = ["A1","A2"]
ch_names = [ch for ch in ch_names if ch not in channels_to_remove]

epocas_list = []
annotations_list = []
eeg_list = []
eventos_list = []
started_sesion_time = []
bad_channels_list = [] ##lista de los canales que se eliminaron en cada run. Usaremos esto para fusionar los canales malos y eliminarlos antes de la concatenación

## CARAGMOS Y FILTRAMOS LOS DATOS DEL O LA SUJETO A ESTUDIAR
##Datos del sujeto y la sesión
n_sujeto = 1
sesion = 1 #1 ejecutado, 2 imaginado
rootpath = "datasets\\"
sujeto = f"sujeto_{n_sujeto}\\"
tarea = "ejec" if sesion == 1 else "imag" ##tarea ejecutada o imaginada
for i in range(0,2):
    run = i+1#1 ##NÚMERO DE RUN 1 o 2
    eeg_file = f"sujeto{n_sujeto}_{tarea}_{run}.hdf5"
    event_file = f"eventos_{tarea}_{run}.txt"

    ##cargamos archivo y lo dejamos de la forma canales x muestras
    data = h5py.File(rootpath+sujeto+eeg_file, "r")
    raweeg = data["RawData"]["Samples"][:,:62].swapaxes(1,0) #descartamos canales A1 y A2

    ##cargo los eventos marcados por el g.HIAMP
    started_time=np.astype(data["AsynchronData"]["Time"][:][0].reshape(-1), int)/sfreq
    total_time = raweeg.shape[1]/sfreq
    started_sesion_time.append([started_time,total_time])
    events_time_ghiamp = np.astype(data["AsynchronData"]["Time"][:][1:].reshape(-1), int)/sfreq
    ##cargo los eventos generados por la app nuestra
    eventos_app = pd.read_csv(rootpath+sujeto+event_file)
    clases = eventos_app["className"].values

    ###Creación de un Montage para el posicionamiento de los electrodos
    montage = mne.channels.read_custom_montage("codes\\ghiamp_montage.sfp")

    noisy_eeg_data = makeRawData(raweeg, sfreq, channel_names=ch_names, montage=montage,
                        event_times=events_time_ghiamp, event_labels=clases)

    ##corto la señal en events_time_ghiamp[0] -3 segundos
    noisy_eeg_data.crop(events_time_ghiamp[0]-3)

    ## ************************ CARGAMOS ICA ENTRENADO Y EL ARCHIVO CSV CON INFORMACIÓN DE PREPROCESAMIENTO ************************
    root_path = f"datasets\\{sujeto}"
    preproc_file = f"preprocessinfo_sujeto_{n_sujeto}_sesion{sesion}_run{run}.csv"
    index = f"Run{run}_TipoSesion{sesion}"
    preproc_file = pd.read_csv(root_path+preproc_file, index_col=0)

    ica_file = f"datasets\\{sujeto}ICA_{eeg_file.split(".")[0]}.fif"
    ##cargamos el archivo ICA
    ica = read_ica(ica_file)
    ica.exclude = [int(comp) for comp in ica.exclude] ##ICA ya tiene almanecados los componentes a eliminar

    ## Cargamos info para eliminar canales y componentes de ICA
    bad_channels = list(preproc_file.loc[index,"bad_channels"].split("-"))
    bad_channels_list.append(bad_channels)
    ##descartamos canales en noisy_eeg_data
    # noisy_eeg_data.drop_channels(bad_channels)

    ###Aplicamos ICA a la señal
    eeg_data_reconstructed = noisy_eeg_data.copy()
    eeg_data_reconstructed = ica.apply(eeg_data_reconstructed)
    eeg_list.append(eeg_data_reconstructed)
    del noisy_eeg_data

## 2. ************************ CONCATENANDO DATOS ************************
## Una vez que hemos cargado y filtrado los datos con ICA, estamos en condiciones de concatenar como si fuera un solo registro

## Evaluamos los canales a eliminar para este sujeto en base a nuestro análisis previo (ver procedimiento en cleaningData.py)
total_bad_channels = []
for bads in bad_channels_list:
    for ch in bads:
        if ch not in total_bad_channels:
            total_bad_channels.append(ch)
print(f"Canales a eliminar: {total_bad_channels}")

eeg_concatenados = mne.concatenate_raws(eeg_list)
eeg_concatenados.drop_channels(total_bad_channels)

eeg_cleaned = eeg_concatenados.copy().filter(l_freq=1.0, h_freq=40, fir_design='firwin', skip_by_annotation='edge')

## 3. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = -0.5, 1.5
event_ids = dict(IZQUIERDA=1, DERECHA=2)
epocas_concatenadas = mne.Epochs(eeg_cleaned, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True)

raw_eventos = mne.events_from_annotations(eeg_cleaned, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

epocas_concatenadas.plot(scalings = 40,show=True, block=True,
                          events=eventos,
                          event_id=event_ids,
                          event_color=dict(IZQUIERDA="red", DERECHA="blue"))

selected_channels = ["C3","C1","Cz","C2","C4"]

epocas_to_analyze = epocas_concatenadas.pick(selected_channels)

## 4. ************************ VARIABLES ÚTILES PARA OBTENER Y ANALIZAR LOS ERDS ************************ 

freqs = np.arange(5, 30)  #rango de frecuencias a estudiar
vmin, vmax = -1, 1.5  # seteamos los valores min y max para el gráfico de ERDS
baseline = (-0.5, -0.1)  # intervalo de la línea de base (en segundos)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, centro max de los ERDS

kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask")

## 5. ************************ COMPUTAMOS LA TIME-FREQUENCY REPRESENTATION (TFR) ************************ 

tfr = epocas_concatenadas.compute_tfr(
    method="multitaper",
    freqs=freqs,
    n_cycles=freqs,
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
    picks=selected_channels)

tfr.crop(-0.5, 2).apply_baseline(baseline, mode="percent")

tfr[2].plot_topo()

selected_channels = ["C1","Cz","C2"]
index_selected_chs = [epocas_concatenadas.ch_names.index(name) for name in selected_channels]

for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(
        1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
    )
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, index_selected_chs[ch]], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, index_selected_chs[ch]], tail=-1, **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot(
            [index_selected_chs[ch]],
            cmap="RdBu",
            cnorm=cnorm,
            axes=ax,
            colorbar=False,
            show=False,
            mask=mask,
            mask_style="mask",
        )

        ax.set_title(index_selected_chs[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if index_selected_chs[ch] != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({event})")
    plt.show()