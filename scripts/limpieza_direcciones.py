from utils.columnas import cargar_datos_activos, contar_y_ultimo_elemento
#from utils.limpieza_direccion import reemplazar, eliminar_info_adicional, ult_palabra, eliminar_cero_si_sigue_numero
from utils.limpieza_direccion import preparar_para_geocoding

import pandas as pd
import unidecode

dir_maestro = r"..\scripts\data\raw\maestro"
n_maestro, lista_maestro = contar_y_ultimo_elemento(dir_maestro)

ultimo_maestro = sorted(lista_maestro)[-1]

data = cargar_datos_activos(dir_maestro + "\\" + ultimo_maestro)

export_dir = r'..\scripts\data\direcciones'
n, lista = contar_y_ultimo_elemento(export_dir)

# Se excluye población a la que no aplica el análisis
data = data[~((data['GRUPO PLANEACION'] == "GRUPO_A") | (data['DESCRIPCION CECO SAP'] == 'Centro Atencion Cliente') | (data['LUGAR DE TRABAJO'] == 'OFICINA CENTRAL APOYO'))]

# Se eliminan registros que 
na =data[data['079_DIRECCION'].isna()]
data.dropna(subset=["079_DIRECCION"], inplace= True)

# Limpieza direcciones
columnas_direcciones = ["NUMERO DE IDENTIFICACION", "079_DIRECCION", "CIUDAD RESIDENCIA", 'CIUDAD DONDE TRABAJA']
data = data[columnas_direcciones]

# data['079_DIRECCION'] = data['079_DIRECCION'].str.lower().apply(unidecode.unidecode).str.replace(r'\s+', ' ', regex= True)

# data['079_DIRECCION'] = data['079_DIRECCION'].apply(reemplazar)
# data['079_DIRECCION'] = data['079_DIRECCION'].apply(eliminar_cero_si_sigue_numero)
# data['079_DIRECCION'] = data['079_DIRECCION'].apply(eliminar_info_adicional)

# data['CIUDAD RESIDENCIA'] = data['CIUDAD RESIDENCIA'].str.lower().apply(unidecode.unidecode).str.replace(r'\s+', ' ', regex= True).replace("bogota d.c.", "bogota")

# data = ult_palabra(data)

data = preparar_para_geocoding(data, '079_DIRECCION', "CIUDAD RESIDENCIA")

data['direccion_geocode'] = data['direccion_geocode'] + ", " + "Colombia"

# data = data[["NUMERO DE IDENTIFICACION", "079_DIRECCION"]]
data.to_csv(export_dir + rf'\direcciones_{n}.csv', index= False)
