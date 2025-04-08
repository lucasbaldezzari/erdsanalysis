"""
Analizando datos en épocas o trials

Se sigue el tutorial de la siguiente página:
https://mne.tools/stable/auto_tutorials/epochs/20_visualize_epochs.html
"""

from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
import h5py
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import read_ica

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

for i in range(0,2):

    ##Datos del sujeto y la sesión
    n_sujeto = 1
    run = i+1#1 ##NÚMERO DE RUN 1 o 2
    sesion = 1 #1 ejecutado, 2 imaginado
    rootpath = "datasets\\"
    sujeto = f"sujeto_{n_sujeto}\\"
    tarea = "ejec" if sesion == 1 else "imag" ##tarea ejecutada o imaginada
    eeg_file = f"sujeto{n_sujeto}_{tarea}_{run}.hdf5"
    event_file = f"eventos_{tarea}_{run}.txt"

    ##cargamos archivo y lo dejamos de la forma canales x muestras
    data = h5py.File(rootpath+sujeto+eeg_file, "r")
    raweeg = data["RawData"]["Samples"][:,:62].swapaxes(1,0) #descartamos canales A1 y A2
    print(raweeg.shape)

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
    ##ploteamos el montage
    # montage.plot(show_names=True)

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
    bad_channels = ["FP1","FP2","AF7","AF8","T7","T8"]
    ##descartamos canales en noisy_eeg_data
    noisy_eeg_data.drop_channels(bad_channels)

    ###Aplicamos ICA a la señal
    eeg_data_reconstructed = noisy_eeg_data.copy()
    eeg_data_reconstructed = ica.apply(eeg_data_reconstructed)
    eeg_list.append(eeg_data_reconstructed)
    del noisy_eeg_data

## ************************ CONCATENANDO EEGse los dos runs ************************
eeg_concatenados = mne.concatenate_raws(eeg_list)
eeg_cleaned = eeg_concatenados.copy().filter(l_freq=1.0, h_freq=40, fir_design='firwin', skip_by_annotation='edge')
# eeg_cleaned.set_eeg_reference(ref_channels=['Cz'])

# new_channels_to_drop = ["O1","O2","PO7","PO3","POz","PO4","PO8","Oz","P7","P8"]
# eeg_cleaned.drop_channels(new_channels_to_drop,"ignore")

# eeg_cleaned.get_montage().plot(show_names=True)

## ************************ SEPARANDO EN ÉPOCAS ************************
##epching de eeg_concatenados
tmin, tmax = -0.5, 2
event_ids = dict(IZQUIERDA=1, DERECHA=2)
epocas_concatenadas = mne.Epochs(eeg_cleaned, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.5, tmax=tmax+0.5,
                                 baseline=None, preload=True)

raw_eventos = mne.events_from_annotations(eeg_cleaned, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

##ploteamos

plotEEG(eeg_cleaned,scalings = 40,show=True, block=True,
        duration = 30, remove_dc = True, bad_color = "red", title="EEGs concatenados",
        event_id=event_ids,
        event_color=dict(IZQUIERDA="red", DERECHA="blue"))

epocas_concatenadas.plot(scalings = 40,show=True, block=True,
                          events=eventos,
                          event_id=event_ids,
                          event_color=dict(IZQUIERDA="red", DERECHA="blue"))

epocas_concatenadas["IZQUIERDA"].plot(scalings = 40,show=True, block=True,
                                      events=eventos, event_id=event_ids,
                                      event_color=dict(IZQUIERDA="red", DERECHA="blue"))

epocas_concatenadas["DERECHA"].plot(scalings = 40,show=True, block=True,
                                      events=eventos, event_id=event_ids,
                                      event_color=dict(IZQUIERDA="red", DERECHA="blue"))

## ************************ TRIALS TO REMOVE ************************
trials_to_remove = [0,8,22]
epocas_concatenadas.drop(trials_to_remove, reason="Bad trial", verbose=True)

## ************************ GRÁFICO DE PSD ************************
epocas_concatenadas.plot_psd(show=True,picks=["C1","C3","C2","C4"],fmax=40,proj=True)
epocas_concatenadas["IZQUIERDA"].plot_psd(show=True,picks=["C1","C3","C2","C4"],fmax=40,proj=True)
epocas_concatenadas["DERECHA"].plot_psd(show=True,picks=["C1","C3","C2","C4"],fmax=40,proj=True)

bands = {"Mu":(8,12), "Beta":(12,30)}
epocas_concatenadas["DERECHA"].plot_psd_topomap(show=True,fmax=40,proj=True,tmin=0,tmax=1,bands=bands,cmap="RdBu_r",colorbar=False)

epocas_concatenadas["DERECHA"].plot_image(picks=["C1","C2"], cmap="RdBu_r")

##graficos de los canales
# epocas_concatenadas.plot_sensors(kind="3d", ch_type="all")
# epocas_concatenadas.plot_sensors(kind="topomap", ch_type="all",show_names=True)

