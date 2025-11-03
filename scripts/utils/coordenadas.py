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
    
    # Distancia al lugar de trabajo actual
    distancia_actual = tiendas_df.loc[tiendas_df['LUGAR DE TRABAJO'] == actual, 'distancia']
    distancia_actual = distancia_actual.iloc[0] if not distancia_actual.empty else None

    # Filtrar tiendas más cercanas que la tienda actual (excluyendo la tienda actual)
    tiendas_mas_cercanas = tiendas_df[
        (tiendas_df['LUGAR DE TRABAJO'] != actual) &
        (tiendas_df['distancia'] < distancia_actual)
    ].sort_values('distancia').reset_index(drop=True)

    # Obtener lista de (LUGAR DE TRABAJO, distancia) de las tiendas más cercanas
    tiendas_mas_cercanas_info = tiendas_mas_cercanas[['LUGAR DE TRABAJO', 'distancia']]
    tiendas_mas_cercanas_lista = list(tiendas_mas_cercanas_info.itertuples(index=False, name=None))

    return pd.Series({
        'distancia_tienda_actual': distancia_actual,
        'tiendas_mas_cercanas_que_actual': tiendas_mas_cercanas_lista if len(tiendas_mas_cercanas_lista) > 0 else None
    })