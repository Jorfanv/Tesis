import pandas as pd
import numpy as np

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import os

# Modelos
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score, auc, roc_curve, precision_recall_curve, log_loss

# función para concatenar conjuntos de datos 
def concatenar(dir: str, sheet: str =None, col: str =None ) -> pd.DataFrame:
    '''
    La siguiente función se encarga de concatenar conjuntos de datos de excel que están en un directorio y 
    tienen el mismo formato, si los conjuntos de datos no tienen el mismo formato la función no 
    funcionará de forma correcta.

    dir --> Directorio

    sheet --> Hoja de calculo en la que está el conjunto de datos si aplica

    col --> Columna que se va a concatenar si así se desea
    '''
    # Archivos 
    files = [f for f in os.listdir(dir) if f.endswith('.xlsx')]

    # Dataframe que almacenará la información de todos los archivos
    dfs = []

    for file in files:
        file_path = os.path.join(dir, file)
        if sheet:
            df = pd.read_excel(file_path, sheet_name=sheet)
        else:
            df = pd.read_excel(file_path)
        if col:
            df = df[[col]]
        dfs.append(df)

    # Se concatenan todos los conjuntos de datos
    return pd.concat(dfs, ignore_index=True)

'''------------------------------------------------------------------------------------------------------------------'''
# Función para análizar columnas 
def analizar_columna(df: pd.DataFrame, columna: str, table : bool = True, plot : bool = False):
    '''
    Analiza una columna específica de un DataFrame y proporciona información detallada.

    Parámetros: 
    df (pd.DataFrame): Dataframe que contiene la columna a analizar.
    columna (str): Nombre de la columna a analizar.

    Retorna:
    dict: Diccionario con información de la columna
    '''
    if columna not in df.columns:
        return{'Error':'La columna no está en el DataFrame'}
    
    info = {}
    info['Nombre'] = columna
    info['Tipo de datos'] = df[columna].dtype
    info['Porcentaje de valores nulos'] = df[columna].isnull().mean() * 100

    if np.issubdtype(info['Tipo de datos'], np.number):
        # Si la columna es numérica, obtener estadísticas descriptivas
        if table == True:
            info['Estadística descriptivas'] = df[columna].describe().to_dict()

        # Gráficos necesarios
        fig, ax = plt.subplots(2, 1, figsize = (8, 6), gridspec_kw = {'height_ratios' : [1, 4]})
        # Boxplot
        sns.boxplot(df[columna].dropna(), color = '#f4fd39', ax = ax[0], orient = 'h')
        ax[0].set_title(f'Boxplot de {columna}', fontsize = 12)
        #ax[0].set_xlabel(columna, fontsize = 10)
        ax[0].grid(axis = 'x', linestyle = '--', alpha = 0.7)

        # Histograma
        sns.histplot(df[columna].dropna(), bins = 30, kde = True, color = 'blue', ax = ax[1])
        ax[1].set_title(f'Histograma de {columna}', fontsize = 12)
        ax[1].set_xlabel(columna, fontsize = 10)
        ax[1].set_ylabel('Frecuencia', fontsize = 10)
        ax[1].grid(axis = 'x', linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        plt.show()
        print(info)

    else:
        # Si la columna es categórica, obtener el conteo por categoría
        #info['Categorías'] = df[columna].unique().tolist()
        #info['Conteo por categoría'] = df[columna].value_counts().to_dict()
        info['Porcentaje por categoría'] = (df[columna].value_counts(normalize=True) * 100).round(2)
        info['Porcentaje acumulado'] = info['Porcentaje por categoría'].cumsum()

        info = pd.DataFrame(info)

        if len(info) < 25:
            plot = True

        if plot == True:
            if len(info) < 5:
                # Gráfico de pastel
                plt.figure(figsize=(6, 4))
                plt.pie(info['Porcentaje por categoría'], labels = info.index, autopct = '%1.1f%%', startangle = 140, colors = sns.color_palette('viridis', len(info)))
                plt.title(f'Gráfico de pastel de {columna}', fontsize = 12)
                plt.show()
            else:
                # Gráfico de barras
                plt.figure(figsize=(6, 4))
                sns.barplot(x = df[columna]. value_counts().index, y = df[columna].value_counts().values, palette = 'viridis')
                plt.title(f'Gráfico de barras de {columna}', fontsize = 12)
                plt.xlabel(columna, fontsize = 10)
                plt.ylabel('Frecuencia', fontsize = 10)
                plt.xticks(rotation = 80, fontsize = 8)
                plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
                plt.show()
        if table == True:
            print(info)

'''------------------------------------------------------------------------------------------------------------------'''

# Creación de funciones para la lectura de los directorios y la actualización de conjuntos de datos 
def obtener_ultimo_archivo(directorio, extension = 'xlsx', posicion = 0):
    ''' Obtiene el último o el archivo por posicion de ultimo en adelante en el directorio que tenga la extensión especificada. '''
    archivos = [archivo for archivo in os.listdir(directorio) if archivo.endswith(f'.{extension}')]
    if not archivos:
        return None
    archivos.sort(key= lambda x: os.path.getctime(os.path.join(directorio, x)), reverse = True)
    return os.path.join(directorio, archivos[posicion])


'''------------------------------------------------------------------------------------------------------------------'''

# Renombrar columnas duplicadas
def rename_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df
'''------------------------------------------------------------------------------------------------------------------'''

def actualizar_datos(directorio, archivo_maestro, columna_id = None, sheet = None):
    '''Actualización del conjunto de datos maestro.'''
    # Obtener el último archivo subido
    archivo_nuevo = obtener_ultimo_archivo(directorio)
    if not archivo_nuevo:
        print('No hay nuevos archivos para unir al archivo maestro')
        return
    print(f'Procesando {archivo_nuevo}')
    df_maestro = pd.read_excel(archivo_maestro)
    df_nuevo = pd.read_excel(archivo_nuevo)
    # Para el conjunto de datos de promociones
    if archivo_maestro ==  '.\\conjuntos_de_datos\\promociones.xlsx':
        df_nuevo.columns = df_nuevo.iloc[0]
        df_nuevo = df_nuevo.iloc[1:]
        df_nuevo = df_nuevo.reset_index(drop = True)
        df_nuevo = rename_duplicate_columns(df_nuevo)
    if sheet:
        df_nuevo = pd.read_excel(archivo_nuevo, sheet_name= sheet)
        # Para el conjunto de datos de Préstamos
        if sheet == 'Préstamos':
            df_nuevo.columns = df_nuevo.iloc[1]
            df_nuevo = df_nuevo.iloc[2:]
            df_nuevo = df_nuevo.reset_index(drop = True)
            df_nuevo = df_nuevo.iloc[:, : -4]
            # Se cambia el nombre de la columna CC para poder concatenar los conjuntos de datos
            df_nuevo = df_nuevo.rename(columns= {'CC': 'Número de Identificación'})
        if type(sheet) == list:
            df_nuevo = pd.read_excel(archivo_nuevo, sheet_name= None)
            df_nuevo_v2 = []
            for i in sheet:
                df = df_nuevo[i]
                if i == 'Auxilios':
                    df.columns = df.iloc[2]
                    df = df.iloc[3:]
                else:
                    df.columns = df.iloc[1]
                    df = df.iloc[2:]
                df = df.iloc[:,:-4]
                df = df.rename(columns= {'CC': 'Número de Identificación'})
                df_nuevo_v2.append(df)
            df_nuevo = pd.concat(df_nuevo_v2, ignore_index = True)
       
    # Eliminar registros que tienen más del 90% de NaN
    threshold = int(df_nuevo.shape[1] * 0.1)
    df_nuevo = df_nuevo.dropna(thresh=threshold)
    # Si el achivo maestro está vacío simplemente se agrega el archivo nuevo
    if df_maestro.empty:
        df_nuevo.to_excel(archivo_maestro, index = False)
        print(f'Se ha creado e archivo maestro con {len(df_nuevo)} registros.')
        return
    
    # Solo mantener las columnas que existen en el archivo maestro
    columnas_maestro = df_maestro.columns 
    df_nuevo = df_nuevo[columnas_maestro.intersection(df_nuevo.columns)] # columnas en común
    # Se convierte el tipo de columna de df_nuevo para que tenga el mismo que el df_master
    for columna in df_maestro.columns:
        df_nuevo[columna] = df_nuevo[columna].astype(df_maestro[columna].dtype, errors = 'ignore')

    # Verificar registros nuevos
    df_nuevo_filtrado = df_nuevo.merge(df_maestro, how = 'left', indicator = True)

    # Filtrar solo las filas que NO están en el df_maestro
    df_nuevo_filtrado = df_nuevo_filtrado[df_nuevo_filtrado['_merge'] == 'left_only'].drop(columns = ['_merge'])

    # Si no hay registros nuevos, no se hace cambios
    if df_nuevo_filtrado.empty:
        print('No hay registros nuevos que agregar.')
        return
    
    # Concatenar y guardar el archivo actualizado 
    df_actualizado = pd.concat([df_maestro, df_nuevo_filtrado], ignore_index = True)
    df_actualizado.to_excel(archivo_maestro, index = False)

    print(f'Archivo actualizado.\n Registros añadidos: {len(df_nuevo_filtrado)}\n Registros totales {len(df_actualizado)}.')

'''------------------------------------------------------------------------------------------------------------------'''
#### Análisis Bivariado de Datos 

### Análisis bibariado de datos numéricos
def analisis_bivariado_numerico(data : pd.DataFrame, var1 : str, var2 : str):
    """
    Realiza un análisis bivariado para variables numéricas.
    
    :param data: DataFrame con los datos
    :param var1: Nombre de la primera variable numérica
    :param var2: Nombre de la segunda variable numérica
    """
    # Diagrama de dispersión 
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=var1, y=var2, data=data, color='blue', alpha=0.5)

    # Personalización del gráfico
    plt.title(f'Diagrama de Dispersión de {var1} y {var2}', fontsize=12, fontweight='bold')
    plt.xlabel(f'{var1}', fontsize=10)
    plt.ylabel(f'{var2}', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()
    # Correlación de Spearman porque las variables numéricas no son normales
    corr, p_value = stats.spearmanr(data[var1], data[var2])
    print(f'Correlación de Spearman: {corr}, p-valor: {p_value}')

### Análisis bivariado de variables categóricas 
def analisis_bivariado_categorico(data : pd.DataFrame, var1 : str, var2 : str, table : bool = False):
    """
    Realiza un análisis bivariado para variables categóricas.
    
    :param data: DataFrame con los datos
    :param var1: Nombre de la primera variable categórica
    :param var2: Nombre de la segunda variable categórica
    """
    # Tabla de contingencia
    contingency_table = pd.crosstab(data[var1], data[var2])
    if table == True:
        print(contingency_table)

    # Test de Chi-cuadrado
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))
    if cramers_v >= 0.5:
        print(f'Prueba Chi-cuadrado de las variables {var1} y {var2}')
        print(f'Chi-cuadrado: {chi2}, p-valor: {p}, Cramer´s V: {cramers_v}')
    

### Análisis bivariado de variables numéricas y categóricas
def analisis_bivariado_mixto(data : pd.DataFrame, var_cat : str, var_num : str):
    """
    Realiza un análisis bivariado para variables mixtas (una categórica y una numérica).
    
    :param data: DataFrame con los datos
    :param var_cat: Nombre de la variable categórica
    :param var_num: Nombre de la variable numérica
    """
    # Boxplot
    plt.figure(figsize=(16, 6))
    sns.boxplot(
        x=var_cat, 
        y=var_num, 
        data=data, 
        width=0.6,  
        linewidth=1, 
        boxprops=dict(facecolor="gray", edgecolor="black"),  
        medianprops=dict(color="red", linewidth=2),  
        whiskerprops=dict(color="black", linewidth=1.2), 
        capprops=dict(color="black", linewidth=1.2),  
    )

    # Titulo 
    plt.title("Distribución de {} por {}".format(var_num, var_cat), fontsize=12, fontweight="bold", pad=15)
    plt.xlabel(var_cat, fontsize=10)
    plt.ylabel(var_num, fontsize=10)
    plt.xticks(rotation=85, fontsize=8)  # Rotar etiquetas del eje X si son largas
    plt.yticks(fontsize=8)
    plt.show()

    # Se verifica la cantidad de categorías para definir la prueba a usar 
    cats = data[var_cat].unique()
    num_cat = len(data[var_cat].unique())

    if num_cat == 2:
        # Distintos grupos del conjuntos de datos 
        gr1 = data[data[var_cat] == cats[0]][var_num]
        gr2 = data[data[var_cat] == cats[1]][var_num]
        # Prueba de U Mann-Whitney U
        u_stat, p_value = stats.mannwhitneyu(gr1, gr2)
        print(f'Prueba de Mann-Whitney U: U-valor = {u_stat}, p-valor = {p_value}')
    else:
        groups = [data[data[var_cat] == cat][var_num] for cat in cats]

        # Pureba de Kruskal-Wallis
        kruskal_result = stats.kruskal(*groups)
        print(f'Prueba de Kruskal-Wallis: H-valor = {kruskal_result.statistic}, p-valor = {kruskal_result.pvalue}')

'''------------------------------------------------------------------------------------------------------------------'''
# Evaluación de modelos
def eval_model (modelo, y_val, y_pred, y_pred_proba = np.array([0]), curva_roc: bool = False, curva_pr: bool = False):
    """
    Evalúa el rendimiento de un modelo de clasificación binaria mediante métricas clave, 
    visualización de la matriz de confusión y curvas ROC/PR opcionales.

    Parámetros:
    ----------
    modelo : sklearn.base.BaseEstimator
        Modelo de clasificación entrenado.
    y_val : array-like
        Valores reales de la variable objetivo (etiquetas verdaderas).
    y_pred : array-like
        Predicciones del modelo (etiquetas predichas).
    y_pred_proba : array-like, opcional
        Probabilidades predichas por el modelo para la clase positiva. Por defecto, un array vacío.
    curva_roc : bool, opcional
        Si es True, genera y muestra la curva ROC junto con el AUC. Por defecto, False.
    curva_pr : bool, opcional
        Si es True, genera y muestra la curva Precision-Recall junto con el AUC. Por defecto, False.

    Retorna:
    -------
    tuple
        Una tupla con las métricas calculadas: (accuracy, precision, recall, f1, specificity).

    Métricas calculadas:
    --------------------
    - Accuracy: Proporción de predicciones correctas.
    - Precision: Proporción de verdaderos positivos entre los positivos predichos.
    - Recall (Sensibilidad): Proporción de verdaderos positivos entre los positivos reales.
    - Specificity (Especificidad): Proporción de verdaderos negativos entre los negativos reales.
    - F1-Score: Media armónica entre precisión y sensibilidad.

    Notas:
    ------
    - Si `curva_roc` o `curva_pr` son True, se requiere que `y_pred_proba` contenga las probabilidades predichas.
    - La función genera gráficos para la matriz de confusión, curva ROC y curva Precision-Recall según corresponda.
    """

    # Se calcula la matriz de confusión modelo base
    matriz_confusion_log = confusion_matrix(y_val, y_pred)

    # Se gráfica la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusion_log, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión modelo Base (Regresión Logística)')
    plt.show()

    ## Métricas importantes del modelo
    # Se extrae los resultados de la matríz de confusión
    tn_log, fp_log, fn_log, tp_log = matriz_confusion_log.ravel()

    # Se calculan las métricas
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='binary')  
    recall = recall_score(y_val, y_pred, average='binary')      
    f1 = f1_score(y_val, y_pred, average='binary')
    specificity = tn_log / (tn_log + fp_log) if (tn_log + fp_log) > 0 else 0 

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensibilidad): {recall:.4f}")
    print(f"Specificity (Especificidad): {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Curva ROC
    if curva_roc == True:
        if len(y_pred_proba) == 1:
            print('Debes ingresar el vector de probabilidades predichas por el modelo.')
        else:
            # Se calcula la curva ROC y se obtiene le valor del AUC
            fpr, tpr, umbrales_roc = roc_curve(y_val, y_pred_proba)

            # Calculo del "mejor" umbral
            diferencias_roc = tpr - fpr # Se busca la diferencia más amplia
            optimal_id_roc = np.argmax(diferencias_roc)
            optimal_roc = umbrales_roc[optimal_id_roc]

            # Calculo del área bajo la curva (AUC)
            roc_auc_log = auc(fpr, tpr)

            # Gráfico
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'Curva ROC (area = {roc_auc_log:.2f})', color='blue')
            plt.scatter(fpr[optimal_id_roc], tpr[optimal_id_roc], color = 'blue', marker='o')
            plt.plot([0, 1], [0, 1], 'r--')  # Línea diagonal (Lo compara con el azar)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Curva ROC - Regresión Logística')
            plt.legend(loc='lower right')
            plt.show()

            print(f"Área bajo la curva (AUC): {roc_auc_log:.4f}")
            print(f"Umbral óptimo: {optimal_roc: .4f}")
    if curva_pr == True:
        if len(y_pred_proba) == 1:
            print('Debes ingresar el vector de probabilidades predichas por el modelo.')
        else:
            # Calculo de la curva PR
            precision_pr, recall_pr, umbrales_pr = precision_recall_curve(y_val, y_pred_proba)

            # Calculo del AUC
            pr_auc_log = auc(recall_pr, precision_pr)

            # Calculo del "mejor" umbral en base a la métrica F1-score
            diferencias_log_pr = 2 * (precision_pr * recall_pr) / (precision_pr + recall_pr)
            optimal_id_pr = np.argmax(diferencias_log_pr)
            optimal_pr = umbrales_pr[optimal_id_pr]

            # Gráfico
            plt.figure(figsize=(8, 6))
            plt.plot(recall_pr, precision_pr, label=f'Curva PR (AUC = {pr_auc_log:.2f})', color = 'blue')
            plt.scatter(recall_pr[optimal_id_pr], precision_pr[optimal_id_pr], color='red', label=f'Umbral óptimo = {optimal_pr:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Cruva Precision-Recall')
            plt.legend()
            plt.grid()
            plt.show()

            print(f"Área bajo la curva (AUC): {pr_auc_log:.4f}")
            print(f"Umbral óptimo: {optimal_pr: .4f}")

    return (accuracy, precision, recall, f1, specificity)

'''------------------------------------------------------------------------------------------------------------------'''
# Grafico de heatmap
def df_heatmap(df, title = "Comparación de métricas de modelos"):
    plt.figure(figsize=(10, 6)) 
    sns.heatmap(df, annot=True, fmt=".3f", cmap="Blues", cbar=True)

    plt.title(f"{title}", fontsize=16)
    plt.xlabel("Modelos", fontsize=12)
    plt.ylabel("Métricas", fontsize=12)

    plt.show()

'''------------------------------------------------------------------------------------------------------------------'''

def deteccion_outliers(df : pd.DataFrame, columna : str) -> pd.DataFrame:
    """
    Detecta outliers en una columna numérica de un DataFrame usando el método IQR.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada.
    columna : str
        Nombre de la columna a analizar.
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con las filas que son outliers en la columna especificada.
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no está en el DataFrame.")
    if not np.issubdtype(df[columna].dtype, np.number):
        raise TypeError(f"La columna '{columna}' no es numérica.")

    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    return outliers
