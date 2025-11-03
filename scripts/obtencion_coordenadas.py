import pandas as pd
import requests
import time
from datetime import date
hoy = date.today()
import os

import warnings
warnings.filterwarnings("ignore")

from utils.columnas import contar_y_ultimo_elemento

# Configuración de directorios
## Directorio que contiene conjunto de datos de direcciones
dir_direcciones = r"..\scripts\data\direcciones"
n_direcciones, lista_direcciones = contar_y_ultimo_elemento(dir_direcciones)

## Directorio que contiene conjunto de datos de coordenadas
dir_coordenadas = r"..\scripts\data\coordenadas"
n_coordenadas, lista_coordenadas = contar_y_ultimo_elemento(dir_coordenadas)

data = sorted(lista_direcciones)[-1]

ultimo_coordenadas = sorted(lista_coordenadas)[-1]
coordenadas_anteriores = pd.read_csv(dir_coordenadas +"\\" + ultimo_coordenadas)
#print(coordenadas_anteriores)

# Obtneción de coordenadas
api_key =123456789

# Rutas de archivos de entrada, back up y salida
dir_entrada = data
dir_salida = dir_coordenadas + rf"\direcciones_geocodificadas{n_coordenadas}.csv"
dir_backup = rf"..\scripts\data\backup_coordenadas\backup_geocodificacion_{hoy}.csv"

# Si ya hay un archivo de salida, retomarlo
if os.path.exists(dir_backup):
    df = pd.read_csv(dir_backup)
    print(f"Cargando proceso existente desde {dir_backup}...")
else:
    df = pd.read_csv(dir_direcciones + "\\" + dir_entrada)

# Se realiza el proceso con los 
comparacion = pd.merge(df, coordenadas_anteriores, on= 'NUMERO DE IDENTIFICACION', how = 'outer', indicator = True)
ambos = comparacion[comparacion['_merge'] == 'both']
izquierda = comparacion[comparacion['_merge'] == 'left_only']

# Limpieza de conjuntos de datos
izquierda.drop(columns=['079_DIRECCION_y', '_merge'], inplace= True)
izquierda.rename(columns= {'079_DIRECCION_x' : '079_DIRECCION'}, inplace= True)
ambos.drop(columns=['079_DIRECCION_x', '_merge'], inplace= True)
ambos.rename(columns= {'079_DIRECCION_y' : '079_DIRECCION'}, inplace= True)

df = izquierda.copy()

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
    elif location_type in ['RANGE_INTERPOLATED', 'GEOMETRIC_CENTER']:
        return 'Media'
    else:
        return 'Baja'

# Procesar las direcciones que no se han procesado
total = len(df)
n = 0
for i, row in df.iterrows():
    n += 1
    direccion = row['079_DIRECCION']
    print(f"[{n}/{total}] Procesando: {direccion}")

    resultado = geocodificar_direccion(direccion)
    df.at[i, 'lat'] = resultado.get('lat')
    df.at[i, 'lng'] = resultado.get('lng')
    df.at[i, 'location_type'] = resultado.get('location_type')
    df.at[i, 'precision'] = clasificar_precision(resultado.get('location_type'))
    df.at[i, 'types'] = resultado.get('types')
    df.at[i, 'status'] = resultado.get('status')
    df.at[i, 'procesado'] = True if resultado.get('status') == 'OK' else False

    if n % 50 == 0:
        print(f"Guardando backup en {dir_backup}...")
        df.to_csv(dir_backup, index= False)
        time.sleep(5)

    time.sleep(0.5)


# Concatenación de conjunto de datos anterior y nuevo 
df = pd.concat([df, ambos], ignore_index= True)

# Resultado final
df.to_csv(dir_salida, index= False)
print(f"\n Proceso completado. Resultados guardados en {dir_salida}")
