import mne
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

eeg_data = concatenateEEGs(sujeto, sesion, apply_ica=True).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")

## 2. ************************ SEPARANDO EN ÉPOCAS ************************

##epOching de eeg_concatenados
tmin, tmax = parameters["duracion_trial"]
event_ids = dict(IZQUIERDA=1, DERECHA=2)
amplitude_rejection = parameters["amplitude_rejection"]
epocas = mne.Epochs(eeg_data, event_id=["IZQUIERDA", "DERECHA"],
                                 tmin=tmin-0.1, tmax=tmax+0.1,
                                 baseline=None, preload=True,reject={"eeg":amplitude_rejection})

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