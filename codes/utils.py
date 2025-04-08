
import xml.etree.ElementTree as ET
import pandas as pd
import mne
import numpy as np

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