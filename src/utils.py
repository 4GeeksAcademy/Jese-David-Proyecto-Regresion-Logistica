import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from IPython.display import display

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

def analisis_categorico(df, target='y', cat_cols=None, plot=False):

        # Detectar columnas categóricas si no se pasan
    if cat_cols is None:
        cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Quitar el target de la lista de categóricas si está
    if target in cat_cols:
        cat_cols = [c for c in cat_cols if c != target]

    print(f"Variables categóricas analizadas: {cat_cols}\n")

    for col in cat_cols:
        print("\n" + "="*80)
        print(f"VARIABLE: {col}")
        print("="*80)

        # ---------- Análisis univariable ----------
        print("\nDistribución univariable (frecuencia y %):")
        conteo = df[col].value_counts(dropna=False)
        porcentaje = df[col].value_counts(normalize=True, dropna=False).mul(100).round(2)
        resumen = pd.DataFrame({'count': conteo, 'percent_%': porcentaje})
        display(resumen)

        if plot:
            conteo.plot(kind='bar')
            plt.title(f'Distribución de {col}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # ---------- Relación con el target (categórica–categórica) ----------
        if target in df.columns:
            print(f"\nRelación {col} vs {target} (tabla de contingencia):")
            tabla = pd.crosstab(df[col], df[target])
            display(tabla)

            print(f"\nRelación {col} vs {target} (porcentaje por fila):")
            tabla_pct = pd.crosstab(df[col], df[target],
                                    normalize='index').mul(100).round(2)
            display(tabla_pct)

            if plot:
                tabla_pct.plot(kind='bar', stacked=True)
                plt.title(f'{col} vs {target} (%)')
                plt.xlabel(col)
                plt.ylabel('% dentro de cada categoría de ' + col)
                plt.xticks(rotation=45)
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
#--------------------------------------------------------------------------------------------------
def outliers_analysis (dataframe, target):    
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numerical_columns = numerical_columns[numerical_columns != target]
    for column in numerical_columns:
        fig, axis = plt.subplots(figsize=(8, 1.2))
        sns.boxplot(ax=axis, data=dataframe, x=column, width=0.3).set(xlabel=None)
        fig.suptitle(column)
        plt.tight_layout()
        plt.show()
    # Return the describe dataframe    
    return dataframe.describe().T
#----------------------------------------------------------------------------------------------------
def outliers_summary(dataset,outliers):
    print(f'''the rows with outliers are {len(outliers)}''')
    print(f'''the total rows are {len(dataset)}''')
    print(f'''this represents {round(len(outliers)/len(dataset),2)*100} % of the dataset''')
#-------------------------------------------------------------------------------------
def small_histogram(dataset, target):
    # columnas numéricas
    numeric_variables = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # quitar el target si está entre las numéricas
    numeric_variables = [col for col in numeric_variables if col != target]

    n_vars = len(numeric_variables)
    n_cols = 3
    n_rows = math.ceil(n_vars / n_cols)   # las filas que hagan falta

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 4*n_rows))

    # convertir axes en array 1D para no liarnos con índices [i//3, i%3]
    axes = np.array(axes).reshape(-1)

    for i, variable in enumerate(numeric_variables):
        ax = axes[i]
        sns.histplot(dataset[variable], bins=20, kde=True, ax=ax)
        ax.set_title(variable)

    # ocultar ejes vacíos si sobran
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()