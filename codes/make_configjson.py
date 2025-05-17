#Archivo para guardar el diccionario de configuracion y características de procesamiento y análisis
#en un archivo json
import json

parameters ={
    "electrodos": ["C3", "C4"],
    "cluster_electrodos_izq": ["C3", "CP3", "C1", "CP1","FC1","FC3"],
    "cluster_electrodos_der": ["C4", "CP4", "C2", "CP2","FC2","FC4"],
    "duracion_trial": [-3, 4],
    "banda_completa": [7, 32],
    "banda_mu": [8, 13],
    "banda_beta": [13, 28],
    "window_size": 512,
    "smoothing_window": 51,
    "baseline_rest": (-1.5, -0.5),
    "baseline_pretask": (-0.5, 0),  # Intervalo de tiempo para el baseline
    "baseline_task": (0, 1),
    "baseline_postask": (2, 3),
    "mean_trials": True,
    "apply_smooth": True,
    "amplitude_rejection": 80, ##en microvolts
    "cmap_topomaps": "RdBu_r",
    "colores_clases": ["#526edc", "#2a8d85"], #IZQUIERDA, DERECHA
}

##guaradamos
with open('codes\\parameters.json', 'w') as f:
    json.dump(parameters, f, indent=4)