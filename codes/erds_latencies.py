import mne
import numpy as np
import matplotlib.pyplot as plt
from codes.utils import concatenateEEGs, getHilbertERDS
from codes.utils import Baseline
import json
import os
import pandas as pd

## 1. ******* CARGAMOS Y CONCATENAMOS LOS DATOS PARA EL/LA SUJETO EN CUESTIÓN *******
##cargo codes\\parameters.json
with open('codes\\parameters.json', 'r') as f:
    parameters = json.load(f)

sujetos = [1]#[1,2,4,5,6,7,8,9]#parameters["sujeto"]
sesiones = [1,2]#parameters["sesion"]

for sujeto in sujetos:
    for sesion in sesiones:
        tipo_sesion = "Ejecutada" if sesion == 1 else "Imaginada"
        sfreq = 512
        channels_to_drop = parameters["channels_to_drop"]
        pick = parameters["pick"]
        confidence = parameters["confidence"]

        ##para mostrar y guardar gráficos
        show = parameters["show_figures"]
        save = parameters["save_figures"]

        ## folder a donde guardar los gráficos
        root_path = os.path.join("datasets", f"sujeto_{sujeto}","info")
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        data = concatenateEEGs(sujeto, sesion, runs=[1,2]).drop_channels(channels_to_drop, "ignore")#.pick(pick,"ignore")
        bandas = parameters["banda_mu"], parameters["banda_beta"]
        names = ["mu", "beta"]
        dict_dfs_bandas = {"mu":None, "beta":None}
        for name_banda, banda in zip(names, bandas ):
            eeg_concatenados = data.copy()
            # banda = parameters["banda_mu"]
            l_freq, h_freq = banda
            eeg_concatenados.filter(l_freq=l_freq, h_freq=h_freq,
                    picks="eeg", 
                    method="fir", 
                    phase="zero-double", 
                    fir_window="hamming",
                    filter_length="auto")
            ## 2. ************************ SEPARANDO EN ÉPOCAS ************************

            ##epOching de eeg_concatenados
            tmin, tmax = parameters["duracion_trial"]
            event_ids = dict(IZQUIERDA=1, DERECHA=2)
            amplitude_rejection = parameters["amplitude_rejection"]
            epocas = mne.Epochs(eeg_concatenados, event_id=["IZQUIERDA", "DERECHA"],
                                            tmin=tmin-0.1, tmax=tmax+0.1,
                                            baseline=None, preload=True,reject={"eeg":amplitude_rejection})

            chs_names = parameters["electrodos"]
            chizq_n = chs_names[0] #nombre del canal izquierdo
            chder_n = chs_names[1] #nombre del canal izquierdo
            chizq_i = [eeg_concatenados.ch_names.index(chs_names[0])] #índice canal izquierdo
            chder_i = [eeg_concatenados.ch_names.index(chs_names[1])] #índice canal derecho
            cluster_izq = [eeg_concatenados.ch_names.index(ch) for ch in parameters["cluster_electrodos_izq"]]
            cluster_der = [eeg_concatenados.ch_names.index(ch) for ch in parameters["cluster_electrodos_der"]]
            clase_izquierda = epocas["IZQUIERDA"]
            clase_derecha = epocas["DERECHA"]
            clase_izquierda_avg = clase_izquierda.average()
            clase_derecha_avg= clase_derecha.average()

            ## 3. ************************ CURVAS ERDS% USANDO HILBERT ************************
            # Graficar curva ERDS para un canal específico (por ejemplo, C3)
            # Intervalo de tiempo para el baseline
            baseline_rest = parameters["baseline_rest"]  # Intervalo de tiempo para el baseline
            baseline_pretask = parameters["baseline_pretask"]  # Intervalo de tiempo para el baseline
            baseline_task = parameters["baseline_task"]
            baseline_postask = parameters["baseline_postask"]
            baseline = Baseline(tuple(baseline_rest))
            ws = 256
            erds_izq = getHilbertERDS(clase_izquierda, baseline, apply_smooth=True, window_smoothing=ws, mean_trials=True)
            erds_der = getHilbertERDS(clase_derecha, baseline, apply_smooth=True, window_smoothing=ws, mean_trials=True)
            erds_izq_avg = erds_izq[cluster_der].mean(0)
            erds_der_avg = erds_der[cluster_izq].mean(0)

            times = clase_izquierda_avg.times
            # df = pd.DataFrame(columns=["min_task","latmin_task","max_task","latmax_task",
            #                            "min_postask","latmin_postask","max_postask","latmax_postask"])
            df = pd.DataFrame(columns=["min_task","latmin_task","max_task","latmax_task",
                                        "min_postask","latmin_postask","max_postask","latmax_postask"],
                                        data = np.zeros((1,8)))
            dict_df = {f"izq_{name_banda}":df.copy(), f"der_{name_banda}":df.copy()}
            for clase, erds in zip(dict_df.keys(),[erds_izq_avg, erds_der_avg]):
                for momento, base in zip(["task","postask"],[baseline_task, baseline_postask]):
                    ti, tf = base
                    indexes = np.where((times>=ti) & (times<=tf))
                    min, max = erds[indexes].min(), erds[indexes].max()
                    lat_min = times[indexes][np.argmin(erds[indexes])]
                    lat_max = times[indexes][np.argmax(erds[indexes])]
                    dict_df[clase][f"min_{momento}"] = min
                    dict_df[clase][f"latmin_{momento}"] = lat_min
                    dict_df[clase][f"max_{momento}"] = max  
                    dict_df[clase][f"latmax_{momento}"] = lat_max

            ##guardo dataframe
            dict_dfs_bandas[name_banda] = pd.concat(dict_df)

        df_final = pd.concat(dict_dfs_bandas)
        ##guardo dataframe
        ##redondeo a dos cifras
        df_final = df_final.round(2)
        df_final.to_csv(os.path.join(root_path, f"lats_suj{sujeto}_ses{sesion}.csv"), index=True)