"""
La idea de este script es analizar y limpiar los datos de EEG para cada sujeto.

Se realizará un análisis usando ICA para ver que componentes son ruido y cuáles son señales de interés.

Se guardará esta información para cada sujeto en un archivo .csv para su posterior análisis.

También se guardarán el modelo ICA para usarlo luego en el procesamiento antes de analizar los datos.
"""

###IMPORTAMOS LAS LIBRERÍAS NECESARIAS

from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
import h5py
import numpy as np
import pandas as pd
import mne

## ************************ 1. PRIMEROS PASOS ************************
"""
Generamos variables que nos permitiran cargar los datos de EEG y los eventos generados por la app.
Los datos de EEG están en un archivo .hdf5 y los eventos generados por la app están en un archivo .txt.
"""
sfreq = 512 # Frecuencia de muestreo
##cargamos los nombres de los electrodos del g.HIAMP
montage_df = pd.read_csv("codes\\ghiamp_montage.sfp",sep="\t",header=None)
ch_names = list(montage_df[0])
channels_to_remove = ["A1","A2"]
ch_names = [ch for ch in ch_names if ch not in channels_to_remove]

##Datos del sujeto y la sesión
n_sujeto = 3
run = 1 ##NÚMERO DE RUN 1 o 2
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

## ************************ 2. PRIMERA INSPECCIÓN PARA EVALUAR QUITAR CANALES ************************
"""
En esta sección se inspecciona la señal para ver si hay canales ruidosos o con mala impedancia. Simplemente
se plotea la señal completa para ver qué canales se pueden eliminar.
"""
##ploteamos la señal completa para ver si hay canales ruidosos debido a una mala impedancia
plotEEG(noisy_eeg_data, scalings = 40,show=True, block=True,
        duration = 20, start = 0, remove_dc = True, bad_color = "red",
        highpas=1, lowpass=40, title="Original filtrada en 1-40Hz para analisis de rechazo de canales")

##Luego de la inspección se decide eliminar los siguientes canales:
bad_channels = []

noisy_eeg_data.drop_channels(bad_channels, "ignore") ##removemos los canales que no sirven

# eeg_data.get_montage().plot(show_names=True) #por si queremos ver el montage con los canales eliminados

## ************************ 3. FILTRANDO EN FRECUENCIA ************************

## NOTA: Ica funciona mejor con señales filtradas con un pasa altos
## Sacado de: https://mne.tools/stable/auto_examples/preprocessing/muscle_ica.html#removing-muscle-ica-components
eeg_data = noisy_eeg_data.copy().filter(l_freq=1.0, h_freq=None, fir_design='firwin', skip_by_annotation='edge')

## ************************ 4. SEPARANDO EN ÉPOCAS ************************
"""
IMPORTANTE: Los eventos fueron marcados en la fase_cue de la app de presentación de estímulos.

Separar en épocas es importante para poder analizar las señales por trial. Esto permite observar cómo se comporta la señal
en cada trial y compararlos entre sí.
"""
##tiempos promedio por trial según la app y los eventos marcados por el g.HIAMP
print(eventos_app["cueInitTime"].diff().mean())
print(np.diff(events_time_ghiamp).mean())

tmin, tmax = -0.5, 2
event_ids = dict(IZQUIERDA=1, DERECHA=2)
epocas = mne.Epochs(eeg_data, event_id=["IZQUIERDA", "DERECHA"],
                    tmin=tmin-0.5, tmax=tmax+0.1,
                    baseline=None, preload=True)


## ************************ 5. APLICANDO ICA PARA ANALIZAR LOS COMPONENTES ************************
"""
Referencia: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
Repairing artifacts with ICA
"""
##entrenamos el ICA para separar los componentes
ica = getICA(eeg_data, n_components = 30, method="fastica")

"""
5.1 ANALIZANDO LOS COMPONENTES
Para analizar los componentes, podemos plotearlos para la señal completa o para las épocas.
Se generan 4 listas las cuales contienen los índices de los componentes que se consideran artefactos,
pero también los que se consideran ritmos alfa en la región occipital.

La idea es graficar las fuentes completas y también las fuentes de las épocas. Esto último permite, en ocasiones, discriminar
si un componente es un artefacto o un ritmo alfa (occipital) o mu (sensorimotor) u o otro ritmo de interés.
"""

ica.plot_components()###ploteo para ver los componentes

ica.plot_sources(eeg_data, title = "Fuentes completas", picks=None)##ploteo para la señal completa
ica.plot_sources(epocas, title="Epocas completas")##ploteo para las epocas
ica.plot_sources(epocas["IZQUIERDA"], title = "Sólo épocas IZQUIERDA")##ploteo para las epocas izquierda
ica.plot_sources(epocas["DERECHA"], title = "Sólo épocas DERECHA")##ploteo para las epocas derecha
ica.plot_sources(eeg_data, picks=[17,18,23,28])
muscle_exclude = ica.find_bads_muscle(eeg_data)[0]
eog_exclude = [1]
ecg_exclude = []
other_exclude = [1,11,17,18,23,25,19,20,27]
dudosos_exclude = []

alpha_occipital = []
erd_possible = []

"""
5.2 ANALIZANDO LAS PROPIEDADES DE LOS COMPONENTES
Podemos plotear las properties para la señal completa o para las epocas.

Esta fase nos sirve para evaluar en mayor profundidad aquellos componentes que se consideran artefactos o ritmos de interés.
"""

# ica.plot_properties(eeg_data, picks=muscle_exclude, psd_args={'fmax': 100.}, image_args={'sigma': 1.})
ica.plot_properties(epocas, picks=muscle_exclude, psd_args={'fmax': 100.}, image_args={'sigma': 1.})

remove_idx_muscle = []##para remover los componentes que NO
muscle_exclude = [compt for compt in muscle_exclude if compt not in remove_idx_muscle]

##luego de una inspección adicional se deciden agregar algunos componentes a la lista de exclusión
muscle_exclude += []

to_exclude = muscle_exclude+eog_exclude+ecg_exclude+alpha_occipital

##usamos overlay para ver que tanto se modifica la señal al eliminar los componentes
ica.plot_overlay(eeg_data, exclude=to_exclude) 

"""
5.3 ELIMINANDO LOS COMPONENTES
"""
ica.exclude = to_exclude

## ************************ 6. RECONSTRUYENDO LA SEÑAL ************************

##reconstruimos la señal de eeg pero filtrada entre 1 y 40Hz y sin los componentes excluidos
eeg_data_reconstructed = noisy_eeg_data.drop_channels(bad_channels,"ignore").copy().filter(l_freq=1.0, h_freq=40., fir_design='firwin', skip_by_annotation='edge')
eeg_data_reconstructed = ica.apply(eeg_data_reconstructed)

##genero las épocas de la señal reconstruida
epocas_reconstructed = mne.Epochs(eeg_data_reconstructed, event_id=["IZQUIERDA", "DERECHA"],
                    tmin=tmin-0.5, tmax=tmax+0.1,  
                    baseline=None, preload=True)

epocas_reconstructed = ica.apply(epocas_reconstructed)

"""
6.1 ANALIZANDO LA SEÑAL RECONSTRUÍDA
"""
### Graficamos el antes y después de aplicar ICA
plotEEG(eeg_data,scalings = 40,show=True, block=True,
    duration = 30, remove_dc = True, bad_color = "red", title="Original filtrada pasa altos")

plotEEG(eeg_data_reconstructed,scalings = 40,show=True, block=True,
    duration = 30, remove_dc = True, bad_color = "red", title="Reconstruida")

"""
6.2 ANALIZANDO LAS ÉPOCAS RECONSTRUÍDAS
"""
### Analizando por épocas
annotations = mne.Annotations(onset=events_time_ghiamp,  ## Eventos en segundos
                                duration=[0] * len(events_time_ghiamp), 
                                description=clases)

raw_eventos = mne.events_from_annotations(eeg_data_reconstructed, event_id=event_ids)
eventos=mne.pick_events(raw_eventos[0], include=[1,2])

##ploteamos las epocas completas
epocas_reconstructed.plot(scalings = 40,show=True, block=True,
                          events=eventos,
                          event_id=event_ids,
                          event_color=dict(IZQUIERDA="red", DERECHA="blue"))

epocas_reconstructed["DERECHA"].plot(scalings = 40,show=True, block=True,
                                     events=eventos,
                                     event_id=event_ids,
                                     event_color=dict(IZQUIERDA="red", DERECHA="blue"))

epocas_reconstructed["IZQUIERDA"].plot(scalings = 40,show=True, block=True,
                                     events=eventos,
                                     event_id=event_ids,
                                     event_color=dict(IZQUIERDA="red", DERECHA="blue"))

# epocas_reconstructed["IZQUIERDA"][0,5,6].plot(scalings = 40,show=True, block=True,
#                                      events=eventos,
#                                      event_id=event_ids,
#                                      event_color=dict(IZQUIERDA="red", DERECHA="blue"))

### *********************** 7. TRIALS A ELIMINAR ************************
"""
La inspección en el punto 6 podría dar lugar a eliminar trials que no son de interés o que son ruido.
"""
trials_to_remove = []

# epocas_reconstructed.drop(trials_to_remove, reason="Ruido") ##eliminamos los trials que no sirven

### *********************** 9. GUARDANDO EL MODELO ICA ************************
##guardamos el modelo ICA
ica.save(f"datasets\\{sujeto}ICA_{eeg_file.split(".")[0]}.fif",overwrite=True)

## *********************** 10. GUARDANDO INFO DE CANALES Y COMPONENTES DESCARTADOS ************************
root_path = f"datasets\\{sujeto}"
df_file = f"preprocessinfo_sujeto_{n_sujeto}_sesion{sesion}_run{run}.csv"
##chequeo si el archivo ya existe en la carpeta
import os
if os.path.exists(root_path+df_file):
    df = pd.read_csv(root_path+df_file,index_col=0)
else:
    df = pd.DataFrame(columns=["sujeto","eeg_file","bad_channels",
                               "bad_muscle_components",
                               "bad_eog_components",
                               "alpha_components",
                               "ecg_components",
                               "trials_to_remove"])

##agregamos la info al dataframe
index = f"Run{run}_TipoSesion{sesion}"

formatted_channels = '-'.join([f"{ch}" for ch in bad_channels])
muscle_exclude_formatted = '-'.join([f"{compt}" for compt in muscle_exclude])
eog_exclude_formatted = '-'.join([f"{compt}" for compt in eog_exclude])
alpha_occipital_formatted = '-'.join([f"{compt}" for compt in alpha_occipital])
ecg_component_formatted = '-'.join([f"{compt}" for compt in ecg_exclude])
formatted_muscle_components = '-'.join([f"{compt}" for compt in muscle_exclude])
formatted_trials_to_remove = '-'.join([f"{trial}" for trial in trials_to_remove])

df.loc[index] = [sujeto,eeg_file,formatted_channels,
                 muscle_exclude_formatted,
                 eog_exclude_formatted,
                 alpha_occipital_formatted,
                 ecg_component_formatted,
                 formatted_trials_to_remove]
##guardamos el dataframe
df.to_csv(root_path+df_file, index=True)