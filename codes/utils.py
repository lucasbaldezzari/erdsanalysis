
import xml.etree.ElementTree as ET
import pandas as pd
import mne
import numpy as np
import h5py
from neuroiatools.EEGManager.RawArray import makeRawData
from mne.preprocessing import read_ica

def xml_to_sfp(xml_path, sfp_path):
    # Leer y parsear el archivo XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extraer los nombres y coordenadas
    names = root.find('electrodename').text.split(',')
    x_positions = list(map(float, root.find('xposition').text.split(',')))
    y_positions = list(map(float, root.find('yposition').text.split(',')))
    z_positions = list(map(float, root.find('zposition').text.split(',')))

    # Crear DataFrame con la estructura deseada
    df = pd.DataFrame({
        'Channel': names,
        'X': x_positions,
        'Y': y_positions,
        'Z': z_positions
    })

    # Guardar el DataFrame en formato SFP (sin encabezados, separado por tabulación)
    df.to_csv(sfp_path, sep='\t', header=False, index=False, float_format='%.6f')

if __name__ == "__main__":
    # Ejemplo de uso
    xml_path = 'codes\\gHIamp_64ch.xml'      # Cambia por la ruta de tu archivo XML
    sfp_path = 'codes\\ghiamp_montage.sfp'  # Cambia por el nombre del archivo SFP de salida

    xml_to_sfp(xml_path, sfp_path)


def applyLaplaciano(raw, center_channel, neighbor_channels, new_channel_name=None):
    """
    Aplica un filtro Laplaciano común a un canal especificado y .

    Parámetros
    ----------
    raw : mne.io.Raw
        Objeto Raw con datos EEG ya cargados.
    center_channel : str
        Nombre del canal al que se le aplicará el filtro Laplaciano.
    neighbor_channels : list of str
        Lista de nombres de canales vecinos a usar para el filtro Laplaciano.
    new_channel_name : str or None
        Si se especifica, se agregará un nuevo canal con ese nombre.
        Si es None, se sobrescribirá el canal original.

    Retorna
    -------
    raw_laplacian : mne.io.Raw
        Objeto Raw con el canal Laplaciano aplicado (como reemplazo o canal nuevo).
    """

    # Verificar que los canales existan
    all_channels = [center_channel] + neighbor_channels
    missing = [ch for ch in all_channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Faltan los siguientes canales en raw: {missing}")
    
    # Obtener datos originales (copiar para no modificar el original in-place)
    raw_laplacian = raw.copy()
    
    # Obtener los datos de los canales involucrados
    data, _ = raw_laplacian.get_data(picks=all_channels, return_times=True)
    
    # Calcular el Laplaciano
    lap_data = data[0] - np.mean(data[1:], axis=0)
    
    if new_channel_name is not None:
        # Crear nuevo canal con el dato Laplaciano
        info = mne.create_info([new_channel_name], sfreq=raw.info['sfreq'], ch_types='eeg')
        lap_raw = mne.io.RawArray(lap_data[np.newaxis, :], info)
        raw_laplacian.add_channels([lap_raw])
    else:
        # Sobrescribir el canal original
        idx = raw_laplacian.ch_names.index(center_channel)
        raw_laplacian._data[idx] = lap_data

    return raw_laplacian

def concatenateEEGs(n_sujeto, sesion, rootpath = "datasets\\",
                    sfreq=512, channels_to_remove = ["A1","A2"]):

    ##cargamos los nombres de los electrodos del g.HIAMP
    montage_df = pd.read_csv("codes\\ghiamp_montage.sfp",sep="\t",header=None)
    ch_names = list(montage_df[0])
    
    ch_names = [ch for ch in ch_names if ch not in channels_to_remove]
    eeg_list = []

    ## CARAGMOS Y FILTRAMOS LOS DATOS DEL O LA SUJETO A ESTUDIAR
    ##Datos del sujeto y la sesión
    
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
        preproc_file = pd.read_csv(root_path+preproc_file, index_col=0)

        ica_file = f"datasets\\{sujeto}ICA_{eeg_file.split(".")[0]}.fif"
        ##cargamos el archivo ICA
        ica = read_ica(ica_file)
        ica.exclude = [int(comp) for comp in ica.exclude] ##ICA ya tiene almanecados los componentes a eliminar
        ##descartamos canales en noisy_eeg_data
        # noisy_eeg_data.drop_channels(bad_channels)

        ###Aplicamos ICA a la señal
        eeg_data_reconstructed = noisy_eeg_data.copy()
        eeg_data_reconstructed = ica.apply(eeg_data_reconstructed)
        eeg_list.append(eeg_data_reconstructed)
        del noisy_eeg_data

    return mne.concatenate_raws(eeg_list)