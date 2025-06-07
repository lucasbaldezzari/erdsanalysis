import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from codes.utils import reacomodar_datos
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel, wilcoxon

sfreq = 512
root_path = "datasets\\latencias"

#cargo datasets\latencias\full_data.csv
full_data = pd.read_csv(os.path.join(root_path, "full_data.csv"))
print(full_data.columns)

save = True
show = False

## 1. ********************** GRAFICOS ****************************
## 1.1 ********************* BOXPLOTS ****************************
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

colores = ["#b16ae7","#fff47c"]#sns.color_palette("Greens", 2)

##Graficamos ERDS
plot_boxplot(
    data=erds_data[(erds_data["Tipo"] == "min") & (erds_data["Banda"] == "mu")],
    x='Período',
    y='ERDS%',
    hue='Sesión',
    title=r'Distribución de ERDS% (mínimos) banda $mu$',
    show=show, save=save,
    filename=f'{root_path}\\erds_distribution_minimos_mu',
    paleta=colores,)

plot_boxplot(
    data=erds_data[(erds_data["Tipo"] == "min") & (erds_data["Banda"] == "beta")],
    x='Período',
    y='ERDS%',
    hue='Sesión',
    title=r'Distribución de ERDS% (mínimos) banda $beta$',
    show=show, save=save,
    filename=f'{root_path}\\erds_distribution_minimos_beta',
    paleta=colores,legend_loc='upper right')

##Graficamos ERDS
plot_boxplot(
    data=erds_data[(erds_data["Tipo"] == "max") & (erds_data["Banda"] == "mu")],
    x='Período',
    y='ERDS%',
    hue='Sesión',
    title=r'Distribución de ERDS% (máximos) banda $mu$',
    show=show, save=save,
    filename=f'{root_path}\\erds_distribution_maximos_mu',
    paleta=colores,)

plot_boxplot(
    data=erds_data[(erds_data["Tipo"] == "max") & (erds_data["Banda"] == "beta")],
    x='Período',
    y='ERDS%',
    hue='Sesión',
    title=r'Distribución de ERDS% (máximos) banda $beta$',
    show=show, save=save,
    filename=f'{root_path}\\erds_distribution_maximos_beta',
    paleta=colores,legend_loc='upper right')

## 1.2 ********************* SCATTERPLOTS ****************************

colors = ["#45b27b","#8c0db0"]#sns.color_palette("Greens", 2)

erds_data["latencias"] = latencias_data["latencias"]
##guardo erds_data con las latencias
erds_data.to_csv(os.path.join(root_path, "erds_data_acomodada.csv"), index=False)

def plot_scatter(data, filtro, x, y, col, row, hue, markers, height=4, aspect=1.5,
                 show=True, save=False, filename=None, palette=None):
    """
    Función para crear un scatterplot con seaborn.
    """
    g = sns.lmplot(
    data=data[data["Tipo"]==filtro],
    x=x, y=y,
    col=col, row=row,
    hue=hue, markers=markers,
    facet_kws={'margin_titles': True},
    height=height, aspect=aspect,
    scatter_kws={'alpha': 0.5, 's': 70},
    palette=palette, sharex=False, sharey=False, ci=None)
    g.axes[0,0].set_xlim(-0.1, 2.1)
    g.axes[0,1].set_xlim(-0.1, 2.1)
    g.axes[1, 0].set_xlim(1.9, 4.1)
    g.axes[1, 1].set_xlim(1.9, 4.1)

    for ax in g.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax.tick_params(axis='both', labelsize=10, top=True, right=True, left=True, bottom=True)
        ax.set_title(ax.get_title(), fontsize=11)

    if save:
        g.savefig(filename, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

##plot para erds min
plot_scatter(
    data=erds_data,
    filtro='min',
    x='latencias',
    y='ERDS%',
    col='Banda',
    row='Período',
    hue='Sesión',
    markers=['o', 's'],
    height=4, aspect=1.5,
    show=show, save=save,
    filename=f'{root_path}\\scatter_erds_min',
    palette=colors)

##plot para erds max
plot_scatter(
    data=erds_data,
    filtro='max',
    x='latencias',
    y='ERDS%',
    col='Banda',
    row='Período',
    hue='Sesión',
    markers=['o', 's'],
    height=4, aspect=1.5,
    show=show, save=save,
    filename=f'{root_path}\\scatter_erds_max',
    palette=colors)

## 1.3 ********************* HEATMAPS **************************** 

heatmap_erds_min = erds_data[erds_data['Tipo'] == 'min'].pivot_table(
    index=['Sujeto'], columns=['Sesión', 'Clase', 'Banda', 'Período'], values='ERDS%')

heatmap_erds_min[('ejecutada','derecha','mu','task')]

def generar_heatmap(df, medida_nombre, save=False, show=True, cmap="coolwarm", figsize=(12, 7)):
    df_pivot = df.pivot_table(
        index="Sujeto",
        columns=["Clase", "Banda", "Período"],
        values="ERDS%",
        aggfunc="min" if medida_nombre == "mínimos" else "max"
    )
    plt.figure(figsize=figsize)
    sns.heatmap(df_pivot, cmap=cmap, center=0, annot=True, fmt=".1f")
    plt.title(f"Heatmap de {medida_nombre} de ERDS% por sujeto y condición")
    plt.xlabel("Condición (Clase, Banda, Período)")
    plt.ylabel("Sujeto")
    plt.tight_layout()
    if save:
        plt.savefig(f"{root_path}\\heatmap_erds_{medida_nombre}.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return df_pivot

heatmap_erds_min_pivot = generar_heatmap(erds_data[erds_data['Tipo'] == 'min'], "mínimos",
                                         cmap="bwr", save=save, show=show)
heatmap_erds_max_pivot = generar_heatmap(erds_data[erds_data['Tipo'] == 'max'], "máximos",
                                         cmap="bwr", save=save, show=show)

## 2 ********************* ANALISIS ESTADÍSTICO **************************** 
## 2.1 ********************* ANOVA ****************************

anova_data = erds_data.copy()
anova_data["Sujeto"] = anova_data["Sujeto"].astype(str)

results = {}
for tipo in ['min', 'max']:
    df_anova = anova_data[anova_data['Tipo'] == tipo].copy()

    # ANOVA de medidas repetidas con cuatro factores
    aovrm = AnovaRM(
        data=df_anova,
        depvar='ERDS%',
        subject='Sujeto',
        within=['Clase', 'Sesión', 'Banda', 'Período']
    )
    results[tipo] = aovrm.fit()

results['min'].summary(), results['max'].summary()

##guardo los resultados de ANOVA en un archivo CSV
indices = results['min'].anova_table.index
columnas = results['min'].anova_table.columns
datos_anova = results['min'].anova_table.values
anova_results = pd.DataFrame(index=indices, columns=columnas, data=datos_anova)
anova_results.to_csv(os.path.join(root_path, "anova_results_min.csv"))

##repito para los resultados de max
datos_anova_max = results['max'].anova_table.values
anova_results_max = pd.DataFrame(index=results['max'].anova_table.index,
                                  columns=results['max'].anova_table.columns,
                                  data=datos_anova_max)
anova_results_max.to_csv(os.path.join(root_path, "anova_results_max.csv"))

## 2.2 ********************* Comparaciones pareadas ****************************

# Almacenar resultados
paired_results = []

# Comparaciones por pares
comparisons = [
    ('Sesión', ['ejecutada', 'imaginada']),
    ('Período', ['task', 'postask']),
    ('Banda', ['mu', 'beta'])
]

# Para cada tipo (ERD / ERS)
for tipo in ['min', 'max']:
    data_tipo = erds_data[erds_data['Tipo'] == tipo]

    for factor, (cond1, cond2) in comparisons:
        # Agrupar por sujeto y condición
        data_pivot = data_tipo.pivot_table(
            index='Sujeto',
            columns=factor,
            values='ERDS%',
            aggfunc='mean'
        )

        # Chequeo: ambas condiciones deben estar presentes
        if cond1 in data_pivot.columns and cond2 in data_pivot.columns:
            x = data_pivot[cond1].dropna()
            y = data_pivot[cond2].dropna()

            # Asegurar igualdad en longitud
            if len(x) == len(y):
                # Prueba t pareada
                t_stat, t_p = ttest_rel(x, y)
                # Wilcoxon (requiere al menos un valor distinto)
                try:
                    w_stat, w_p = wilcoxon(x, y)
                except ValueError:
                    w_stat, w_p = None, None

                paired_results.append({
                    'Tipo': tipo,
                    'Comparación': f"{cond1} vs {cond2}",
                    'Media {0}'.format(cond1): x.mean(),
                    'Media {0}'.format(cond2): y.mean(),
                    't-p': t_p,
                    'Wilcoxon-p': w_p
                })

df_pareadas = pd.DataFrame(paired_results)
# Guardar resultados de comparaciones pareadas
df_pareadas.to_csv(os.path.join(root_path, "comparaciones_pareadas.csv"), index=False)