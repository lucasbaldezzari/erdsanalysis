import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from codes.utils import reacomodar_datos

sfreq = 512
root_path = "datasets\\latencias"

#cargo datasets\latencias\full_data.csv
full_data = pd.read_csv(os.path.join(root_path, "full_data.csv"))
print(full_data.columns)

save = False
show = True


## 1. ********************* BOXPLOTS ****************************
"""
Distribuciones de los valores mínimos y máximos
de ERDS% para cada banda (mu, beta), período (task, postask),
sesión (imag, eje) y clase (izq, der), agrupado por sujeto.
"""

erds_data = pd.concat([
    reacomodar_datos(full_data, banda, extremo, periodo, f"{banda}_{extremo}_{periodo}", "ERDS%")
    for banda in ['mu', 'beta']
    for extremo in ['min', 'max']
    for periodo in ['task', 'postask']])

latencias_data = pd.concat([
    reacomodar_datos(full_data, banda, extremo, periodo, f"{banda}_{extremo}_lat_{periodo}", "latencias")
    for banda in ['mu', 'beta']
    for extremo in ['min', 'max']
    for periodo in ['task', 'postask']])

def plot_boxplot(data, x, y,hue,title, figsize=(8, 6), show=True, save=False, filename=None,
                 paleta=None,legend_loc='lower left'):
    """
    Función para crear un boxplot con seaborn.
    """
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data = data, x = x, y = y, hue = hue,
                     palette=paleta)
    ax.set_title(title, fontsize=14)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.legend(title=hue, loc=legend_loc, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(f"{filename}.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

colors = ["#76edb1","#97C7E6"]#sns.color_palette("Greens", 2)

##Graficamos ERDS
plot_boxplot(
    data=erds_data[erds_data["Tipo"]=="min"],
    x='Período',
    y='ERDS%',
    hue='Sesión',
    title='Distribución de ERDS% (mínimos) por Banda y Período',
    show=show, save=save,
    filename=f'{root_path}\\erds_distribution_minimos',
    paleta=colors,)

##Graficamos ERDS
plot_boxplot(
    data=erds_data[erds_data["Tipo"]=="max"],
    x='Período',
    y='ERDS%',
    hue='Sesión',
    title='Distribución de ERDS% (máximos) por Banda y Período',
    show=show, save=save,
    filename=f'{root_path}\\erds_distribution_maximos',
    paleta=colors,)

## 2. ********************* SCATTERPLOTS ****************************

colors = ["#45b27b","#5b0672"]#sns.color_palette("Greens", 2)

erds_data["latencias"] = latencias_data["latencias"]

plt.figure(figsize=(12, 8))
g = sns.lmplot(
    data=erds_data,
    x="latencias", y="ERDS%",
    col="Período", row="Tipo",
    hue="Sesión", markers=["o", "s"],
    facet_kws={'margin_titles': True},
    height=4, aspect=1.5,
    scatter_kws={'alpha': 0.5, 's': 60},
    palette=colors, sharex=False, sharey=False,
)
g.axes[0,0].set_xlim(-0.1, 2.1)
g.axes[1,0].set_xlim(-0.1, 2.1)
g.axes[0, 1].set_xlim(1.9, 4.1)
g.axes[1, 1].set_xlim(1.9, 4.1)

# for ax in g.axes.flatten():
#     ax.set_xlabel(ax.get_xlabel(), fontsize=12)
#     ax.set_ylabel(ax.get_ylabel(), fontsize=12)
#     ax.tick_params(axis='both', labelsize=10, top=True, right=True, left=True, bottom=True)
    
#     # ax.spines['top'].set_visible(True)
#     # ax.spines['right'].set_visible(True)


## 3. ********************* HEATMAPS **************************** 


heatmap_erds_min = erds_data[erds_data['Tipo'] == 'min'].pivot_table(
    index=['Sujeto'], columns=['Sesión', 'Clase', 'Banda', 'Período'], values='ERDS%')