
import xml.etree.ElementTree as ET
import pandas as pd
import mne
import numpy as np
import h5py
from neuroiatools.EEGManager.RawArray import makeRawData
from mne.preprocessing import read_ica
import scipy.signal as signal
from dataclasses import dataclass

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

def concatenateEEGs(n_sujeto, sesion, rootpath = "datasets\\",montage_file = "codes\\ghiamp_montage.sfp",
                    sfreq=512, channels_to_remove = ["A1","A2"], apply_ica=True):

    ##cargamos los nombres de los electrodos del g.HIAMP
    montage_df = pd.read_csv(montage_file,sep="\t",header=None)
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

        eeg_data = makeRawData(raweeg, sfreq, channel_names=ch_names, montage=montage,
                            event_times=events_time_ghiamp, event_labels=clases)

        ##corto la señal en events_time_ghiamp[0] -3 segundos
        eeg_data.crop(events_time_ghiamp[0]-3)

        ## ************************ CARGAMOS ICA ENTRENADO Y EL ARCHIVO CSV CON INFORMACIÓN DE PREPROCESAMIENTO ************************
        root_path = f"datasets\\{sujeto}"
        preproc_file = f"preprocessinfo_sujeto_{n_sujeto}_sesion{sesion}_run{run}.csv"
        preproc_file = pd.read_csv(root_path+preproc_file, index_col=0)

        if apply_ica:

            ica_file = f"datasets\\{sujeto}ICA_{eeg_file.split(".")[0]}.fif"
            ##cargamos el archivo ICA
            ica = read_ica(ica_file)
            ica.exclude = [int(comp) for comp in ica.exclude] ##ICA ya tiene almanecados los componentes a eliminar
            ##descartamos canales en eeg_data
            # eeg_data.drop_channels(bad_channels)

            ###Aplicamos ICA a la señal
            
            eeg_data = ica.apply(eeg_data)
           
        eeg_list.append(eeg_data)
        del eeg_data

    return mne.concatenate_raws(eeg_list)

def concatenateSubjectsEEG(sujetos, sesion, rootpath = "datasets\\",montage_file = "codes\\ghiamp_montage.sfp",
                    sfreq=512, channels_to_remove = ["A1","A2"], apply_ica=True):
    """
    Sujetos: Lista (enteros) de personas voluntarias
    """

    ##cargamos los nombres de los electrodos del g.HIAMP
    montage_df = pd.read_csv(montage_file,sep="\t",header=None)
    ch_names = list(montage_df[0])
    
    ch_names = [ch for ch in ch_names if ch not in channels_to_remove]
    eeg_list = []

    for n_sujeto in sujetos:
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

            eeg_data = makeRawData(raweeg, sfreq, channel_names=ch_names, montage=montage,
                                event_times=events_time_ghiamp, event_labels=clases)

            ##corto la señal en events_time_ghiamp[0] -3 segundos
            eeg_data.crop(events_time_ghiamp[0]-3)

            ## ************************ CARGAMOS ICA ENTRENADO Y EL ARCHIVO CSV CON INFORMACIÓN DE PREPROCESAMIENTO ************************
            root_path = f"datasets\\{sujeto}"
            preproc_file = f"preprocessinfo_sujeto_{n_sujeto}_sesion{sesion}_run{run}.csv"
            preproc_file = pd.read_csv(root_path+preproc_file, index_col=0)

            if apply_ica:

                ica_file = f"datasets\\{sujeto}ICA_{eeg_file.split(".")[0]}.fif"
                ##cargamos el archivo ICA
                ica = read_ica(ica_file)
                ica.exclude = [int(comp) for comp in ica.exclude] ##ICA ya tiene almanecados los componentes a eliminar
                ##descartamos canales en eeg_data
                # eeg_data.drop_channels(bad_channels)

                ###Aplicamos ICA a la señal
                
                eeg_data = ica.apply(eeg_data)
            
            eeg_list.append(eeg_data)
            del eeg_data

    return mne.concatenate_raws(eeg_list)

def loadOA(n_sujeto, rootpath = "datasets\\",montage_file = "codes\\ghiamp_montage.sfp",
                    sfreq=512, channels_to_remove = ["A1","A2"], apply_ica=True, ica_file="ICA_OA.fif"):
    """
    Carga la señal de Ojos Abiertos
    """

    ##cargamos los nombres de los electrodos del g.HIAMP
    montage_df = pd.read_csv(montage_file,sep="\t",header=None)
    ch_names = list(montage_df[0])
    
    ch_names = [ch for ch in ch_names if ch not in channels_to_remove]
    eeg_list = []

    ## CARAGMOS Y FILTRAMOS LOS DATOS DEL O LA SUJETO A ESTUDIAR
    ##Datos del sujeto y la sesión
    
    sujeto = f"sujeto_{n_sujeto}\\"

    eeg_file = f"sujeto{n_sujeto}_oa.hdf5"

    ##cargamos archivo y lo dejamos de la forma canales x muestras
    data = h5py.File(rootpath+sujeto+eeg_file, "r")
    raweeg = data["RawData"]["Samples"][:,:62].swapaxes(1,0) #descartamos canales A1 y A2

    ##cargo los eventos marcados por el g.HIAMP
    ##cargo los eventos generados por la app nuestra


    ###Creación de un Montage para el posicionamiento de los electrodos
    montage = mne.channels.read_custom_montage(montage_file)

    eeg_data = makeRawData(raweeg, sfreq, channel_names=ch_names, montage=montage)

    if apply_ica:

        ica_file = f"{rootpath}{sujeto}{ica_file}"
        ##cargamos el archivo ICA
        ica = read_ica(ica_file)
        ica.exclude = [int(comp) for comp in ica.exclude] ##ICA ya tiene almanecados los componentes a eliminar
        ##descartamos canales en eeg_data
        # eeg_data.drop_channels(bad_channels)

        ###Aplicamos ICA a la señal
        
        eeg_data = ica.apply(eeg_data)

    return eeg_data

    ##corto la señal en events_time_ghiamp[0] -3 segundos

def getHilbertERDS(eeg_data, baseline, apply_smooth: bool = True, window_smoothing: int = 50, mean_trials = True):
    """
    Retorna datos de ERDS% usando Hilbert
    Parameters:
    ----------
    eeg_data : mne.Epochs
        Objeto Epochs con datos EEG ya cargados.
    baseline : Baseline
        Objeto Baseline con los tiempos de la línea base (tmin, tmax), timpo de suavizado y si se aplica suavizado o no.
    Returns:
    -------
    erds : array_like
        Array con los datos de ERDS%.
    """
    hilbert_data = eeg_data.apply_hilbert(envelope=True)  # Aplicar el filtro de Hilbert
    if mean_trials:
        mean_over_trials = hilbert_data.get_data().mean(axis=0)  # Promedio de la envolvente sobre los trials
        # ti, tf = baseline
        # idx_baseline = np.where((eeg_data.times >= ti) & (eeg_data.times <= tf))[0]
        baseline_mean = baseline.apply(mean_over_trials, hilbert_data.times)  # Calcular la línea base

        erds = 100*(mean_over_trials - baseline_mean) / baseline_mean  ## Calculo el ERDS%
        if apply_smooth:
            # Suavizamos la señal usando una ventana de tamaño window_smoothing
            erds = np.array([np.convolve(channel, np.ones(window_smoothing)/window_smoothing, mode='same') for channel in erds])
            
        return erds
    else:
        erds = hilbert_data.get_data()  # Obtener los datos de la envolvente
        for i, trial in enumerate(hilbert_data):
            baseline_mean = baseline.apply(trial, hilbert_data.times)
            data_baseline = 100*(trial - baseline_mean) / baseline_mean  ## Calculo el ERDS%
        if apply_smooth:
            # Suavizamos la señal usando una ventana de tamaño window_smoothing
            data_baseline = np.array([np.convolve(channel, np.ones(window_smoothing)/window_smoothing, mode='same') for channel in data_baseline])
        erds[i] = data_baseline
        return erds

@dataclass
class Baseline:
    """
    Clase para almacenar y aplicar un baseline a los datos EEG.
    Attributes
    ----------
    baseline : tuple
        Tupla con los tiempos de la línea base (tmin, tmax).
    random_window : bool
        Si es True, se aplicará una ventana aleatoria para el baseline entre los tiempos especificados en window_duration.
    window_duration : tuple
        Tupla con los tiempos de inicio y fin de la ventana aleatoria.
    """
    baseline: tuple
    random_window:bool = False
    window_duration:tuple = None

    def __post_init__(self):
        if not isinstance(self.baseline, tuple) or len(self.baseline) != 2:
            raise ValueError("El atributo 'baseline' debe ser una tupla de longitud 2.")
        ##chequeamos que window_duration sea una tupla de longitud 2, que los tiempos de inicio y fin no esten fuera de rango respecto de los tiempos de baseline
        if self.random_window and (self.window_duration is None or not isinstance(self.window_duration, tuple) or len(self.window_duration) != 2): 
            raise ValueError("El atributo 'window_duration' debe ser una tupla de longitud 2.")
        if self.random_window and (self.window_duration[0] < self.baseline[0] or self.window_duration[1] > self.baseline[1]):
            raise ValueError("Los tiempos de inicio y fin de 'window_duration' deben estar dentro del rango de 'baseline'.")
        
    def apply(self,data, times):
        """
        Aplica el baseline a los datos EEG.
        Parameters
        ----------
        data : array_like shape (n_channels, n_times)
            Datos EEG a los que se aplicará el baseline.
        times : array_like shape (n_times,)
            Tiempos correspondientes a los datos EEG.
        Returns
        -------
        data : array_like
            Datos EEG con el baseline aplicado.
        """
        ti, tf = self.baseline
        ##si random_window es True, seleccionamos una ventana aleatoria dentro de los tiempos de baseline
        if self.random_window:
            ti_random = np.random.uniform(self.window_duration[0], self.window_duration[1])
            tf_random = ti_random + (tf - ti)
            idx_baseline = np.where((times >= ti_random) & (times <= tf_random))[0]
        else:
            idx_baseline = np.where((times >= ti) & (times <= tf))[0]

        return np.mean(data[:,idx_baseline])

def getEpochsBaseline(eeg_data, baseline, times):
    """
    Aplica el baseline a los datos EEG.
    Parameters
    ----------
    eeg_data : mne.Epochs
        Objeto Epochs con datos EEG ya cargados.
    baseline : Baseline
        Objeto Baseline con los tiempos de la línea base (tmin, tmax).
    times : array_like shape (n_times,)
        Tiempos correspondientes a los datos EEG.
    Returns
    -------
    data : array_like
        Datos EEG con el baseline aplicado.
    """
    eeg_baseline = eeg_data.copy()
    for i, trial in enumerate(eeg_data):
        baseline_mean = baseline.apply(trial, times)
        eeg_baseline._data[i,] = (trial - baseline_mean) / baseline_mean

    return eeg_baseline


def compute_stft_power(eeg_signal, sfreq, nperseg=256, noverlap=128):
    """
    Computa la STFT de la señal EEG y devuelve la potencia tiempo-frecuencia.
    Parameters
    ----------
    eeg_signal : array_like
        Señal EEG de entrada.
    sfreq : float
        Frecuencia de muestreo de la señal EEG.
    nperseg : int, optional
        Número de puntos por segmento para la STFT (default es 256).
    noverlap : int, optional
        Número de puntos de solapamiento entre segmentos (default es 128).
    Returns
    -------
    f : array_like
        Frecuencias de la STFT.
    """
    f, t, Zxx = signal.stft(eeg_signal, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2  # Power = |STFT|^2 se calcula la potencia elevando al cuadrado el módulo de la STFT
    return f, t, power

def compute_band_power(power, freqs, band):
    """
    Sum the power in a specific frequency band.
    Suma la potencia en una banda de frecuencia específica.
    Parameters
    ----------
    power : array_like
        Potencia de la STFT (frecuencia x tiempo).
    freqs : array_like
        Frecuencias de la STFT.
    band : tuple
        Banda de frecuencia (min, max) para la cual se desea calcular la potencia.
    Returns
    -------
    band_power : array_like
        Potencia total en la banda de frecuencia especificada.
    """
    band_indices = np.where((freqs >= band[0]) & (freqs <= band[1]))[0] ##filtramos los índices en la banda de interés
    band_power = np.sum(power[band_indices, :], axis=0) ##sumamos la potencia en la banda de interés a lo largo del tiempo
    return band_power

def compute_erd_ers_stft(band_power, baseline_indices, task_indices):
    """
    Compute ERD/ERS (%) based on STFT band power.
    Calcula ERD/ERS (%) basado en la potencia de banda STFT.
    Parameters
    ----------
    band_power : array_like
        Potencia de banda STFT (frecuencia x tiempo).
    baseline_indices : array_like
        Índices de la señal de referencia (línea base).
    task_indices : array_like
        Índices de la señal de tarea (ejecución).
    Returns
    -------
    erd_ers_percent : float
        Porcentaje de ERD/ERS.
    """
    baseline_power = np.mean(band_power[baseline_indices])
    task_power = np.mean(band_power[task_indices])
    erd_ers_percent = ((task_power - baseline_power) / baseline_power) * 100
    return erd_ers_percent

class LaplacianFilter:
    """
    Filtro Laplaciano local para EEG aplicado a eeg_dataetos mne.Raw o mne.Epochs.
    Permite sobrescribir canales o agregar canales nuevos con sufijo "_lap".

    Parameters
    ----------
    montage : dict
        Diccionario con claves de nombre de electrodos y valores con listas de vecinos.
            Ejemplo: 
            montage = {
            'C3': ['C1', 'C5', 'CP3', 'FC3'],
            'C4': ['C2', 'C6', 'CP4', 'FC4']}
    """

    def __init__(self, montage):
        self.montage = montage

    def apply(self, eeg_data, picks=None, inplace=True):
        """
        Aplica el filtro Laplaciano sobre los canales seleccionados.

        Parameters
        ----------
        eeg_data : mne.io.Raw | mne.Epochs
            eeg_data: objeto Raw o Epochs de MNE sobre el cual aplicar el filtro.
        picks : list of str, optional
            Canales a filtrar. Por defecto se filtran todos los definidos en el montaje.
        inplace : bool, optional
            Si True, sobrescribe los datos originales. Si False, agrega nuevos canales con sufijo '_lap'.
        """
        if picks is None:
            picks = list(self.montage.keys())

        data = eeg_data.get_data()  # shape: (n_channels, n_times) o (n_epochs, n_channels, n_times)
        ch_names = eeg_data.ch_names
        is_epochs = data.ndim == 3

        for ch in picks:
            if ch not in self.montage:
                raise ValueError(f"No hay vecinos definidos para el canal {ch}")
            neighbors = self.montage[ch]
            all_required = [ch] + neighbors
            if not all(chan in ch_names for chan in all_required):
                raise ValueError(f"Faltan canales para calcular el Laplaciano de {ch}")

            idx_c = ch_names.index(ch)
            idx_neighbors = [ch_names.index(n) for n in neighbors]

            if is_epochs:
                lap = data[:, idx_c, :] - np.mean(data[:, idx_neighbors, :], axis=1)
            else:
                lap = data[idx_c, :] - np.mean(data[idx_neighbors, :], axis=0)

            if inplace:
                if is_epochs:
                    data[:, idx_c, :] = lap
                else:
                    data[idx_c, :] = lap
            else:
                # Crear nuevo canal
                new_ch_name = f"{ch}_lap"
                ch_type = eeg_data.get_channel_types(picks=ch)[0]
                info = mne.create_info([new_ch_name], eeg_data.info['sfreq'], ch_types=ch_type)
                new_raw = mne.io.RawArray(lap[np.newaxis] if not is_epochs else lap.mean(axis=0, keepdims=True), info)

                if isinstance(eeg_data, mne.io.BaseRaw):
                    eeg_data.add_channels([new_raw], force_update_info=True)
                elif isinstance(eeg_data, mne.Epochs):
                    eeg_data._data = np.concatenate([eeg_data._data, lap[:, np.newaxis, :]], axis=1)
                    eeg_data.info = mne.channels.combine_channels.combine_infos([eeg_data.info, info])

        if isinstance(eeg_data, mne.io.BaseRaw):
            eeg_data._data = data
        elif isinstance(eeg_data, mne.Epochs):
            eeg_data._data = data

class EnvolventeEEG:
    def __init__(self, raw_data, smoothing_window=None):
        """
        Inicializa la clase con el objeto mne.Raw o mne.Epochs y el tamaño de ventana para el suavizado.
        
        :param raw_data: Puede ser un objeto mne.Raw o mne.Epochs.
        :param smoothing_window: Tamaño de la ventana para el suavizado. Si es None, no se aplica suavizado.
        """
        ##chequeamos que el objeto de entrada sea mne.Raw o mne.Epochs
        if not isinstance(raw_data, (mne.io.Raw, mne.Epochs)):
            raise TypeError("El objeto de entrada debe ser mne.Raw o mne.Epochs.")
        ##chequeamos que el tamaño de la ventana sea un entero positivo
        if smoothing_window is not None and (not isinstance(smoothing_window, int) or smoothing_window <= 0):
            raise ValueError("El tamaño de la ventana debe ser un entero positivo.")
        ##chequeamos que el objeto de entrada tenga datos
        if raw_data._data is None:
            raise ValueError("El objeto de entrada no tiene datos.")
        self.data = raw_data.copy()  # Hacemos una copia para no modificar el original
        self.data_envelope = None
        self.smoothing_window = smoothing_window
        
    def _suavizar(self, señal):
        """
        Aplica suavizado (media móvil) a la señal utilizando np.convolve.
        
        :param señal: Array de la envolvente de la señal.
        :return: Señal suavizada.
        """
        if self.smoothing_window is not None:
            ventana = np.ones(self.smoothing_window) / self.smoothing_window  # Ventana de media móvil
            return np.convolve(señal, ventana, mode='same')
        return señal
    
    def procesar(self, db=False):
        """
        Reemplaza la señal de EEG por su envolvente para cada canal, utilizando apply_hilbert() y aplicando
        suavizado si es necesario.
        
        :return: Objeto mne.Raw o mne.Epochs con las señales reemplazadas por sus envolventes.
        """
        if isinstance(self.data, mne.io.Raw):
            # Calculamos la envolvente utilizando apply_hilbert()
            self.data_envelope = self.data.copy().apply_hilbert(envelope=True)
            # Suavizamos la envolvente de cada canal si es necesario
            for ch in range(self.data.info['nchan']):
                self.data_envelope._data[ch, :] = self._suavizar(self.data_envelope._data[ch, :])
            # Si se desea, se puede aplicar una escala en decibelios
            if db:
                self.data_envelope._data = 20 * np.log10(np.abs(self.data_envelope._data))
            return self.data_envelope

        elif isinstance(self.data, mne.Epochs):
            # Calculamos la envolvente utilizando apply_hilbert()
            self.data_envelope = self.data.apply_hilbert(envelope=True)

            # Aseguramos de suavizar cada canal de cada época
            for ch in range(self.data.info['nchan']):
                for epoca in range(self.data._data.shape[0]):  # Recorremos las épocas
                    self.data_envelope._data[epoca, ch, :] = self._suavizar(self.data_envelope._data[epoca, ch, :])
            # Si se desea, se puede aplicar una escala en decibelios
            if db:
                self.data_envelope._data = 20 * np.log10(np.abs(self.data_envelope._data))

            return self.data_envelope
        
def getIntervalos(freqs, paso):
    """
    Función para obtener una lista de lista de intervalos a partir de una de intervalos y un paso.
    Se toman los intervalos y cuando el siguiente intervalo supera el paso, se crea un nuevo intervalo.
    """
    lista_intervalos = []
    if len(freqs) > 0:
        intervalo_actual = [float(np.round(freqs[0],1))]
        for i in range(1, len(freqs)):
            if freqs[i] - intervalo_actual[-1] > paso:
                lista_intervalos.append(intervalo_actual)
                intervalo_actual = [float(np.round(freqs[i],1))] # reiniciamos el intervalo actual
            else:
                intervalo_actual.append(float(np.round(freqs[i],1)))
        lista_intervalos.append(intervalo_actual) # añadimos el último intervalo
        ##ordeno de menor a mayor cada intervalo
        for i in range(len(lista_intervalos)):
            lista_intervalos[i] = sorted(lista_intervalos[i])
    return lista_intervalos