from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
import h5py
import numpy as np
import pandas as pd
import mne
from codes.utils import loadOA

## ************************ 1. PRIMEROS PASOS ************************
"""
Generamos variables que nos permitiran cargar los datos de EEG y los eventos generados por la app.
Los datos de EEG están en un archivo .hdf5 y los eventos generados por la app están en un archivo .txt.
"""
##Datos del sujeto y la sesión
sujeto = 1
root_path = "datasets\\"

bad_channels=[]

eeg_data = loadOA(sujeto)
eeg_data.crop(5)


plotEEG(eeg_data, scalings = 40,show=True, block=True,
        duration = 20, start = 0, remove_dc = True, bad_color = "red",
        highpass=1, lowpass=40, title=f"Señal de sesión Ojos Abiertos para sujeto {sujeto}",)

## 2. ************************ APLICAMOS ICA ************************
ica = getICA(eeg_data, n_components = 30, method="fastica")
ica.plot_components()###ploteo para ver los componentes
ica.plot_sources(eeg_data, title = "Fuentes completas", picks=None)##ploteo para la señal completa

muscle_exclude = ica.find_bads_muscle(eeg_data)[0]
eog_exclude = [0,1]
ecg_exclude = []
other_exclude = []
dudosos_exclude = []

alpha_occipital = []
erd_possible = []

"""
2.2 ANALIZANDO LAS PROPIEDADES DE LOS COMPONENTES
Podemos plotear las properties para la señal completa o para las epocas.

Esta fase nos sirve para evaluar en mayor profundidad aquellos componentes que se consideran artefactos o ritmos de interés.
"""
# ica.plot_properties(eeg_data, picks=muscle_exclude, psd_args={'fmax': 100.}, image_args={'sigma': 1.})

to_exclude = muscle_exclude+eog_exclude+ecg_exclude+alpha_occipital
##usamos overlay para ver que tanto se modifica la señal al eliminar los componentes
ica.plot_overlay(eeg_data, exclude=to_exclude) 

"""
2.3 ELIMINANDO LOS COMPONENTES
"""
ica.exclude = to_exclude

## ************************ 3. RECONSTRUYENDO LA SEÑAL ************************
eeg_reconstructed = ica.apply(eeg_data.copy())

### Graficamos el antes y después de aplicar ICA
plotEEG(eeg_data,scalings = 40,show=True, block=True,
    duration = 30, remove_dc = True, bad_color = "red", title="Original filtrada pasa altos")

plotEEG(eeg_reconstructed,scalings = 40,show=True, block=True,
    duration = 30, remove_dc = True, bad_color = "red", title="Reconstruida")


### *********************** 4. GUARDANDO EL MODELO ICA ************************
##guardamos el modelo ICA
ica.save(f"datasets\\sujeto_{sujeto}\\ICA_OA.fif",overwrite=True)

## *********************** 5. GUARDANDO INFO DE CANALES Y COMPONENTES DESCARTADOS ************************
root_path = f"datasets\\sujeto_{sujeto}\\"
df_file = f"preprocessinfo_sujeto_{sujeto}_OA.csv"
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

# ##agregamos la info al dataframe
index = f"TipoSesion_OjosAbiertos"

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