import pandas as pd
import requests
import time
from datetime import date
import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

api_key = 123456789

# Función para geocodificar las direcciones
def geocodificar_direccion(direccion, max_intentos = 3):
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {
        'address' : direccion,
        'key' : api_key
    }

    for intento in range(max_intentos):
        try:
            respuesta = requests.get(base_url, params= params, timeout= 15)
            if respuesta.status_code != 200:
                print(f"Error HTTP: {respuesta.status_code}")
                time.sleep(2)
                continue
            
            data = respuesta.json()
            
            if data['status'] == 'OK':
                resultados = data['results'][0]
                return {
                    'lat' : resultados['geometry']['location']['lat'],
                    'lng' : resultados['geometry']['location']['lng'],
                    'location_type' : resultados['geometry']['location_type'],
                    'types' : ', '.join(resultados['types']),
                    'status' : 'OK'
                }
            elif data['status'] in ['OVER_QUERY_LIMIT', 'UNKNOWN_ERROR']:
                print(f"Error de la API de Google ({data['status']}), reintentando...")
                time.sleep(3 * (intento + 1))
            else:
                return {'status' : data['status']}
        except Exception as e:
            print(f"Excepción al consultar API: {e}")
            time.sleep(1)
    
    return {'status' : 'Falló'}

# Función para clasificar precisión de la geocodificación
def clasificar_precision(location_type):
    if location_type == 'ROOFTOP':
        return 'Alta'
    elif location_type in ['RANGE_INTERPOLATED']:
        return 'Media'
    else:
        return 'Baja'
    

# Creación conjunto de datos con coordenadas
def creacion_df_coordendas(df: pd.DataFrame, dir_backup: str):
    # Si existe backup, cargarlo y actualizar el df original
    if os.path.exists(dir_backup):
        print(f"Cargando backup existente desde {dir_backup}...")
        df_backup = pd.read_csv(dir_backup)
        # Asegurar que las columnas nuevas existan en el df original
        for col in ['lat', 'lng', 'location_type', 'precision', 'types', 'status', 'procesado']:
            if col not in df.columns:
                df[col] = None
        # Actualizar los registros procesados en el df original
        df.set_index('direccion_geocode', inplace=True)
        df_backup.set_index('direccion_geocode', inplace=True)
        df.update(df_backup)
        df.reset_index(inplace=True)
        df_backup.reset_index(inplace=True)
        df_direcciones = df
    else:
        df_direcciones = df
        # Inicializar columnas si no existen
        for col in ['lat', 'lng', 'location_type', 'precision', 'types', 'status', 'procesado']:
            if col not in df_direcciones.columns:
                df_direcciones[col] = None

    total = len(df_direcciones)
    n = 0
    # Procesar solo los registros no procesados
    for i, row in df_direcciones[df_direcciones['procesado'] != True].iterrows():
        n += 1
        direccion = row['direccion_geocode']
        print(f"[{n}/{total}] Procesando: {direccion}")

        resultado = geocodificar_direccion(direccion)
        df_direcciones.at[i, 'lat'] = resultado.get('lat')
        df_direcciones.at[i, 'lng'] = resultado.get('lng')
        df_direcciones.at[i, 'location_type'] = resultado.get('location_type')
        df_direcciones.at[i, 'precision'] = clasificar_precision(resultado.get('location_type'))
        df_direcciones.at[i, 'types'] = resultado.get('types')
        df_direcciones.at[i, 'status'] = resultado.get('status')
        df_direcciones.at[i, 'procesado'] = True if resultado.get('status') == 'OK' else False

        if n % 50 == 0:
            print(f"Guardando backup en {dir_backup}...")
            df_direcciones.to_csv(dir_backup, index= False)
            time.sleep(5)

        time.sleep(0.5)
    # Guardar backup final
    print(f"Guardando backup final en {dir_backup}...")
    df_direcciones.to_csv(dir_backup, index= False)
    return df_direcciones

    