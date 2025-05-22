# Repo para el analisis de ERDS

## Pre-rocesamiento 

El archivo [cleaningData.py](https://github.com/lucasbaldezzari/erdsanalysis/blob/main/codes/cleaningData.py) describe paso a paso cómo realizar el análisis y la limpieza de datos EEG utilizando técnicas de procesamiento de señales utilizando Python y la librería MNE, centrándose especialmente en la eliminación de artefactos usando ICA. Los pasos que se realizan dentro del script se describen en docs/ica_readme.md-

## Obteniendo información

### Análisis de ERDS usando Hilbert

El script [curvas_erds](https://github.com/lucasbaldezzari/erdsanalysis/blob/main/codes/curvas_erds.py) contiene código para el procesamiento y obtención de curvas $ERDS{\%}$.

### Análisis de ERD y ERS en tiempo frecuencia

El script [erds_timefreq.py](https://github.com/lucasbaldezzari/erdsanalysis/blob/main/codes/erds_timefreq.py) contiene código para procesar y obtener un gráfico de variación del PSD en porcentaje junto con mapas topográficos.

### Diferencias de potencia entre los estados _rest vs tarea_ y _rest vs post tarea_

El script [erds_spectral](https://github.com/lucasbaldezzari/erdsanalysis/blob/main/codes/erds_spectral.py) se utiliza para procesar y obtener unas curvas de la diferencia de potencia entre los estados _rest vs tarea_ y _rest vs post tarea_.