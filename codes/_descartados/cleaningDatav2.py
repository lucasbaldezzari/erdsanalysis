import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs, getHilbertERDS
from codes.utils import Baseline
from neuroiatools.SignalProcessor.ICA import getICA
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

eeg_data = concatenateEEGs(sujeto, sesion, runs=[1,2],apply_ica=True)#.drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")
eeg_data.plot(scalings=40)
## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = parameters["duracion_trial"]
event_ids = dict(IZQUIERDA=1, DERECHA=2)
amplitude_rejection = parameters["amplitude_rejection"]
epocas = mne.Epochs(eeg_data, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True,)

## ************************ 5. APLICANDO ICA PARA ANALIZAR LOS COMPONENTES ************************
"""
Referencia: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
Repairing artifacts with ICA
"""
##entrenamos el ICA para separar los componentes
ica = getICA(eeg_data, n_components = 30, method="fastica")

ica.plot_components(show = False)###ploteo para ver los componentes

ica.plot_sources(eeg_data, title = "Fuentes completas", picks=None,show = False)##ploteo para la señal completa
# ica.plot_sources(epocas, title="Epocas completas",show = False)##ploteo para las epocas
ica.plot_sources(epocas["IZQUIERDA"], title = "Sólo épocas IZQUIERDA",show = False)##ploteo para las epocas izquierda
ica.plot_sources(epocas["DERECHA"], title = "Sólo épocas DERECHA",show = False)##ploteo para las epocas derecha
plt.show()

muscle_exclude = [10,13,18,20,21] #ica.find_bads_muscle(eeg_data)[0]
eog_exclude = [0,1,3,4]
ecg_exclude = []
other_exclude = []
dudosos_exclude = []

alpha_occipital = [7]
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
eeg_data_reconstructed = eeg_data.copy()
eeg_data_reconstructed = ica.apply(eeg_data_reconstructed)

##genero las épocas de la señal reconstruida
epocas_reconstructed = mne.Epochs(eeg_data_reconstructed, event_id=["IZQUIERDA", "DERECHA"],
                    tmin=tmin-0.5, tmax=tmax+0.1,  
                    baseline=None, preload=True)

epocas_reconstructed = ica.apply(epocas_reconstructed)

epocas_reconstructed.plot(scalings=40)

# raw_eventos = mne.events_from_annotations(eeg_data_reconstructed, event_id=event_ids)
# eventos=mne.pick_events(raw_eventos[0], include=[1,2])


# epocas_reconstructed["DERECHA"].plot(scalings = 40,show=True, block=True,
#                                      events=eventos,
#                                      event_id=event_ids,
#                                      event_color=dict(IZQUIERDA="red", DERECHA="blue"))

# epocas_reconstructed["IZQUIERDA"].plot(scalings = 40,show=True, block=True,
#                                      events=eventos,
#                                      event_id=event_ids,
#                                      event_color=dict(IZQUIERDA="red", DERECHA="blue"))

ica.save(f"datasets\\sujeto_{sujeto}\\ICA_sujeto{sujeto}_{tipo_sesion}_EEGConcatenados.fif",overwrite=True)


## *********************** 10. GUARDANDO INFO DE CANALES Y COMPONENTES DESCARTADOS ************************
root_path = f"datasets\\sujeto_{sujeto}\\"
df_file = f"preprocessinfo_sujeto_{sujeto}_sesion{sesion}_eegconcat.csv"
##chequeo si el archivo ya existe en la carpeta
import os
if os.path.exists(root_path+df_file):
    df = pd.read_csv(root_path+df_file,index_col=0)
else:
    df = pd.DataFrame(columns=["sujeto",
                               "bad_muscle_components",
                               "bad_eog_components",
                               "alpha_components",
                               "ecg_components",])

##agregamos la info al dataframe
index = f"TipoSesion{sesion}"

muscle_exclude_formatted = '-'.join([f"{compt}" for compt in muscle_exclude])
eog_exclude_formatted = '-'.join([f"{compt}" for compt in eog_exclude])
alpha_occipital_formatted = '-'.join([f"{compt}" for compt in alpha_occipital])
ecg_component_formatted = '-'.join([f"{compt}" for compt in ecg_exclude])

df.loc[index] = [sujeto,
                 muscle_exclude_formatted,
                 eog_exclude_formatted,
                 alpha_occipital_formatted,
                 ecg_component_formatted]
##guardamos el dataframe
df.to_csv(root_path+df_file, index=True)