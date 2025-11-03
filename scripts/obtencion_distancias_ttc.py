import pandas as pd
import googlemaps
import time
import os

# api_key = "123456789"

# # Cliente de googlemaps
# gmaps = googlemaps.Client(key=api_key)


def distancia_duracion_transporte(origin, destination):
    try:
        result = gmaps.distance_matrix(origin, destination, mode="transit")
        element = result['rows'][0]['elements'][0]
        if element['status'] == 'OK':
            dist_km = element['distance']['value'] / 1000  # metros a km
            dur_min = element['duration']['value'] / 60    # segundos a minutos
            return dist_km, dur_min
    except Exception as e:
        print(f"Error en API: {e}")
    return None, None

def calcular_distancias_transporte_backup(df_empleados, df_tiendas, dir_backup):
    # Si existe backup, cargarlo y filtrar empleados ya procesados
    if os.path.exists(dir_backup):
        print(f"Cargando backup existente desde {dir_backup}...")
        df_backup = pd.read_csv(dir_backup)
        ids_procesados = set(df_backup['id_persona'])
    else:
        df_backup = pd.DataFrame()
        ids_procesados = set()
    
    resultados = []
    n = 0
    total = len(df_empleados)
    for idx, row in df_empleados.iterrows():
        id_persona = row['NÚMERO DE CEDULA'] # Ajuste para el conjunto de retirados
        # id_persona = row['NUMERO DE IDENTIFICACION'] # Ajuste para el conjunto de activos
        if id_persona in ids_procesados:
            continue  # Saltar si ya está en backup

        origen = f"{row['lat']},{row['lng']}"
        actual = row['LUGAR DE TRABAJO']
        tienda_actual = df_tiendas[df_tiendas['LUGAR DE TRABAJO'] == actual]
        if not tienda_actual.empty:
            destino_actual = f"{tienda_actual.iloc[0]['latitud']},{tienda_actual.iloc[0]['longitud']}"
            dist_actual, tiempo_actual = distancia_duracion_transporte(origen, destino_actual)
        else:
            dist_actual, tiempo_actual = None, None

        dict_cercanos = {}
        # Solo calcular para tiendas cercanas si hay distancia y tiempo para el lugar de trabajo actual
        if dist_actual is not None and tiempo_actual is not None:
            cercanos = row['tiendas cercanas']
            if isinstance(cercanos, list):
                for nombre in cercanos:
                    tienda = df_tiendas[df_tiendas['LUGAR DE TRABAJO'] == nombre]
                    if not tienda.empty:
                        destino = f"{tienda.iloc[0]['latitud']},{tienda.iloc[0]['longitud']}"
                        dist, tiempo = distancia_duracion_transporte(origen, destino)
                        dict_cercanos[nombre] = {'distancia_km': dist, 'tiempo_min': tiempo}

        resultados.append({
            'id_persona': id_persona,
            'distancia_actual_km': dist_actual,
            'tiempo_actual_min': tiempo_actual,
            'distancias_cercanos': dict_cercanos
        })
        n += 1

        # Guardar backup cada 10 empleados procesados
        if n % 10 == 0:
            print(f"[{n}/{total}] Guardando backup en {dir_backup}...")
            df_temp = pd.DataFrame(resultados)
            df_final = pd.concat([df_backup, df_temp], ignore_index=True)
            df_final.to_csv(dir_backup, index=False)
            time.sleep(2)
        time.sleep(0.4)
    # Guardar backup final
    print(f"Guardando backup final en {dir_backup}...")
    df_temp = pd.DataFrame(resultados)
    df_final = pd.concat([df_backup, df_temp], ignore_index=True)
    df_final.to_csv(dir_backup, index=False)
    return df_final