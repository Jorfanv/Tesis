import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

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
        return(info)

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
            return(info)
        
"-----------------------------------------------------------------------------------------------------"
# Función para obtener fecha correcta en archivos .xlsb
def obt_fecha (columna):
    columna = pd.to_datetime(columna, origin = '1899-12-30', unit = 'D')
    return columna

def contar_y_ultimo_elemento(ruta):
    elementos = os.listdir(ruta)
    elementos = [e for e in elementos if not e.startswith('.')] 
    cantidad = len(elementos)
    return cantidad, elementos

# Función de lectura de archivo 
@st.cache_data
def cargar_datos():
    dir_data_app = r"..\scripts\data\app"
    n_data_app, lista_data_app = contar_y_ultimo_elemento(dir_data_app)
    ultimo_data_app = sorted(lista_data_app)[-1]
    df = pd.read_csv(dir_data_app + "\\" + ultimo_data_app)
    return df

def cargar_datos_tiendas():
    df = pd.read_csv(r"..\src\data\proccesed\tiendas.csv")
    return df

def intercambios_sugeridos(df):
    df_filtered = df.dropna(subset=['NUMERO DE IDENTIFICACION', 'LUGAR DE TRABAJO', 
                                    'tienda_mas_cercana_diferente', 'DESCRIPCION PUESTO', 'mejora_%'])

    df_filtered['id'] = df_filtered['NUMERO DE IDENTIFICACION']

    candidatos = df_filtered[[
        'id', 'NOMBRE COMPLETO', '079_DIRECCION', 'DESCRIPCION TIEMPO TEORICO','LUGAR DE TRABAJO', 'tienda_mas_cercana_diferente',
        'DESCRIPCION PUESTO', 'mejora_%'
    ]].copy()

    candidatos.columns = [
        'id_1', 'nombre_1', 'direccion_1','jornada_1', 'tienda_actual_1', 'tienda_sugerida_1', 'puesto_1', 'mejora_1'
    ]

    intercambios = pd.merge(
        candidatos, candidatos,
        left_on=['tienda_sugerida_1', 'tienda_actual_1', 'puesto_1','jornada_1'],
        right_on=['tienda_actual_1', 'tienda_sugerida_1', 'puesto_1','jornada_1'],
        suffixes=('_a', '_b')
    )

    intercambios = intercambios[
        (intercambios['id_1_a'] != intercambios['id_1_b']) &
        (intercambios['mejora_1_a'] > 0) &
        (intercambios['mejora_1_b'] > 0)
    ]

    intercambios_resultado = intercambios.rename(columns={
        'id_1_a': 'ID_Empleado_1',
        'nombre_1_a': 'Nombre_1',
        'tienda_actual_1_a': 'Tienda_Actual_1',
        'tienda_sugerida_1_a': 'Tienda_Sugerida_1',
        'mejora_1_a': 'Mejora_1_%',
        
        'id_1_b': 'ID_Empleado_2',
        'nombre_1_b': 'Nombre_2',
        'tienda_actual_1_b': 'Tienda_Actual_2',
        'tienda_sugerida_1_b': 'Tienda_Sugerida_2',
        'mejora_1_b': 'Mejora_2_%',
        
        'puesto_1': 'Cargo'
    })
    return intercambios_resultado

# Lectura de conjuntos de datos
def cargar_datos_activos(dir):
    return (pd.read_excel(dir, sheet_name="Activos", header=1, engine="pyxlsb"))


