import numpy as np
import pandas as pd
# Función para calcular distancia entre dos puntos geográficos
def calcular_distancia(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos geográficos utilizando la fórmula de Haversine.

    Parámetros:
    lat1, lon1: Coordenadas del primer punto (latitud y longitud).
    lat2, lon2: Coordenadas del segundo punto (latitud y longitud).

    Retorna:
    Distancia en kilómetros.
    """
    R = 6371  # Radio de la Tierra en kilómetros
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distancia = R * c
    return distancia

# --------------------------------------------------------------------------------------------------------------------
def analizar_distancias_por_empleado(empleado, tiendas_df):
    lat_emp = empleado['lat']
    lon_emp = empleado['lng']
    actual = empleado['LUGAR DE TRABAJO']
    
    # Calcular distancias a todas las tiendas
    tiendas_df['distancia'] = calcular_distancia(
        lat_emp, lon_emp, tiendas_df['latitud'], tiendas_df['longitud']
    )
    
    # Ordenar por distancia
    tiendas_ordenadas = tiendas_df.sort_values('distancia').reset_index(drop=True)
    
    # Top 3 tiendas con distancias
    top_3_info = tiendas_ordenadas.head(3)[['LUGAR DE TRABAJO', 'distancia']]
    top_3_lista = list(top_3_info.itertuples(index=False, name=None))
    
    # Distancia al lugar de trabajo actual
    distancia_actual = tiendas_df.loc[tiendas_df['LUGAR DE TRABAJO'] == actual, 'distancia']
    distancia_actual = distancia_actual.iloc[0] if not distancia_actual.empty else None

    # Buscar tienda más cercana distinta
    mejor_opcion = None
    for _, row in tiendas_ordenadas.iterrows():
        if row['LUGAR DE TRABAJO'] != actual:
            if distancia_actual is not None and row['distancia'] < distancia_actual:
                mejor_opcion = row['LUGAR DE TRABAJO']
            break

    return pd.Series({
        'tienda_mas_cercana_diferente': mejor_opcion,
        'top_3_tiendas_cercanas': top_3_lista,
        'distancia_tienda_actual': distancia_actual
    })