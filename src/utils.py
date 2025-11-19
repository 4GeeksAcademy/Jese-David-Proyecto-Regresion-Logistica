import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
#------------------------------------------------------------------------------------------
def plot_numerical_data(dataframe):
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    for column in numerical_columns:
        fig, axis = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [6, 1]})

        # Calculate mean, median, and standard deviation
        mean_val = np.mean(dataframe[column])
        median_val = np.median(dataframe[column])
        std_dev = np.std(dataframe[column])

        # Create a multiple subplots with histograms and box plots
        sns.histplot(ax=axis[0], data=dataframe, kde=True, x=column).set(xlabel=None)
        axis[0].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        axis[0].axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        axis[0].axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1, label='Standard Deviation')
        axis[0].axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)

        sns.boxplot(ax=axis[1], data=dataframe, x=column, width=0.6).set(xlabel=None)
        axis[1].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        axis[1].axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        axis[1].axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1)
        axis[1].axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)

        axis[0].legend()

        fig.suptitle(column)

        plt.tight_layout()

        plt.show()
#-----------------------------------------------------------------------------------------------------
def plot_categorics_hist(dataframe):
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns

    n_cols = 3
    n_rows = math.ceil(len(categorical_cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, categorical_cols):
        sns.countplot(data=dataframe, x=col, hue=col, ax=ax, palette='pastel', legend=False)
        ax.set_title(col)
        ax.tick_params(axis='x', rotation=45)

    # Borrar axes vacíos
    for i in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    #----------------------------------------------------------------------------------
def plot_scatter_heatmaps(dataframe, target_variable):
    # columnas numéricas
    numeric_variables = dataframe.select_dtypes(include=['float64', 'int64']).columns

    # quitamos la variable target de la lista de X
    x_variables = [col for col in numeric_variables if col != target_variable]

    num_cols = 2
    num_rows = len(x_variables)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(13, 5 * num_rows))

    # asegurar que axes sea 2D aunque haya una sola fila
    if num_rows == 1:
        axes = np.array([axes])

    for row_idx, x_variable in enumerate(x_variables):
        # Gráfico de dispersión + recta de regresión
        sns.regplot(
            ax=axes[row_idx, 0],
            data=dataframe,
            x=x_variable,
            y=target_variable
        )
        axes[row_idx, 0].set_title(f'Regplot: {x_variable} vs {target_variable}')

        # Heatmap de correlación
        corr = dataframe[[x_variable, target_variable]].corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            ax=axes[row_idx, 1]
        )
        axes[row_idx, 1].set_title(f'Heatmap: {x_variable} vs {target_variable}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
#-------------------------------------------------------------------------------------------------
def general_heatmap(dataframe, target_variable):
    # Seleccionar solo columnas numéricas
    numeric_df = dataframe.select_dtypes(include=['float64', 'int64'])

    # Si el target es categórico y lo has convertido a 0/1, inclúyelo
    if target_variable in dataframe.columns:
        numeric_df[target_variable] = dataframe[target_variable]

    # Reordenar para que el target aparezca primero
    cols = [col for col in numeric_df.columns if col != target_variable]
    reordered = pd.concat([numeric_df[target_variable], numeric_df[cols]], axis=1)

    # Crear heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(reordered.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Heatmap de correlaciones (solo variables numéricas)")
    plt.tight_layout()
    plt.show()