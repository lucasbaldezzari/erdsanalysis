# Repo para el analisis de ERDS

## *cleaningData.py* -> Pre procesamiento 

Se describe paso a paso cómo realizar el análisis y la limpieza de datos EEG utilizando técnicas de procesamiento de señales utilizando Python y la librería MNE, centrándose especialmente en la eliminación de artefactos usando ICA.

### Introducción

El objetivo principal de este script es limpiar y analizar datos de EEG registrados mediante el amplificador g.HIAMP. Para ello, se emplea el análisis por componentes independientes (ICA) que permite identificar y eliminar componentes de ruido como movimientos musculares (artefactos EMG), movimientos oculares (artefactos EOG) y otros artefactos, preservando al máximo las señales cerebrales de interés.

Además, se almacenan tanto el modelo ICA obtenido como un registro detallado de canales y componentes descartados para análisis posteriores.

#### Librerías utilizadas
- neuroiatools: Herramientas específicas para EEG.
- MNE-Python: Biblioteca especializada para EEG/MEG.
- h5py, numpy, pandas: Para manipulación eficiente de datos.

### Pasos del análisis EEG

1. **Carga y Preparación de Datos**

Se definen variables importantes como frecuencia de muestreo (sfreq=512 Hz), archivos de EEG (.hdf5), archivos de eventos (.txt) y posición de electrodos (.sfp).

Se carga el archivo EEG, descartando canales específicos (A1, A2), y los eventos generados por la aplicación externa.

2. **Primera inspección visual y descarte de canales**

Se genera un objeto Raw de MNE con datos originales.

Se realiza una inspección visual inicial para evaluar el estado general de los canales y determinar aquellos con mala impedancia o mucho ruido visual.

Los canales identificados como ruidosos (e.g. "AF7", "AF8", "T7", "T8", "F9") son descartados.

**IMPORTANTE:** Lo que se debe hacer en primer lugar es cargar todas las señales correspondientes a una sesión, por ejemplo los dos runs correspondientes a la sesión 1. Lugo, evaluar visualmente qué canales deben eliminarse para ambos registros, y en base a esto, armar una única lista con los canales a eliminar.

Esta remoción de canales será usada en ambos registros para aplicar el punto 3 en adelante, por cada registro por separado.

3. **Filtrado en Frecuencia**

La señal es filtrada usando un filtro pasa-altos de 1 Hz para mejorar el desempeño del ICA, siguiendo las recomendaciones prácticas de MNE-Python.

4. **Segmentación en Épocas**

Los datos se segmentan en épocas alrededor de los eventos de interés (estímulos presentados), con un intervalo temporal específico (tmin=-1, tmax=3 segundos), facilitando la comparación y análisis de señales por trial.

Los tiempos seleccionados son considerando la duración del cue (2 segundos). Se podrían probar un tmin y tmax diferentes.

5. **Aplicación de ICA (Independent Component Analysis)**

En este apartado:

- Se aplica ICA (fastica) para descomponer la señal EEG en componentes independientes que permiten identificar fuentes de ruido y señales de interés.

- Se visualizan los componentes en gráficos de fuentes completas y segmentadas por clase (izquierda/derecha).

- Se definen listas con índices de componentes que corresponden a diferentes tipos de artefactos:

    - Artefactos musculares (muscle_exclude).

    - Artefactos oculares (eog_exclude).

    - Componentes dudosos o a evaluar más detalladamente (dudosos_exclude).

    - Ritmos alfa occipitales y posibles componentes ERD (Event-Related Desynchronization).

6. **Reconstrucción de la Señal**

Los componentes identificados como ruido son excluidos y se reconstruye la señal EEG filtrada en la banda de 1-40 Hz.

Se crean nuevamente las épocas para comparar antes y después de aplicar ICA.

La señal reconstruida se visualiza nuevamente para validar que se hayan eliminado correctamente los artefactos.

7. **Segunda ronda de eliminación de canales (Opcional)**

Se revisa nuevamente la señal para evaluar si algún canal adicional debe ser descartado.

8. **Eliminación de Trials (Opcional)**

Después de evaluar la señal reconstruida y las épocas individuales, podrían eliminarse trials específicos si se identifican claramente como ruido.

De todos modos, se aconseja dejar este paso para cuando se realice el análisis de ERD/ERS en la señal.

9. **Guardado del modelo ICA**

El modelo ICA entrenado se guarda en formato *.fif* para poder reutilizarlo en futuros análisis, asegurando consistencia en la metodología.

10. **Guardado de información sobre canales y componentes descartados**

Se genera un archivo .csv que registra detalladamente los canales descartados y componentes excluidos, asegurando transparencia y reproducibilidad del análisis.

#### Consideraciones finales

Este pipeline garantiza una limpieza minuciosa y reproducible de los datos EEG, facilitando análisis posteriores tales como clasificación de tareas motoras (movimientos ejecutados vs. imaginados), análisis de patrones espaciales y temporales, o generación de interfaces cerebro-computadora (BCI).

Para ejecutar correctamente este script asegúrate de tener todos los archivos de datos en las rutas indicadas y contar con el entorno Python configurado adecuadamente, especialmente con la librería MNE-Python.

#### Referencias

- [MNE-Python ICA Artifact Correction](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html)
- [MNE-Python Muscle Artifact Removal](https://mne.tools/stable/auto_examples/preprocessing/muscle_ica.html)