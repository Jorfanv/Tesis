import streamlit as st
import pydeck as pdk
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

from utils.columnas import cargar_datos, cargar_datos_tiendas, intercambios_sugeridos, cargar_datos_activos
from filters import sidebar_filters, aplicar_filtros
from layout import mostrar_metricas, mostrar_tabla, mostra_grafica, mostrar_colaborador, colaborador_tienda_sugerida, intercambio
from config import mostrar_banner
from map_layers import crear_capa_tiendas, crear_capa_colaboradores, crear_arcos, crear_capa_tiendas_sugeridas, crear_arcos_tienda_sugerida

st.set_page_config(layout='wide', page_title='Geolocalización', page_icon='🌍')

# Mostrar banner
mostrar_banner("GEOLOCALIZACIÓN Y REUBICACIÓN DE COLABORADORES")

# Cargar datos
georref = cargar_datos()
tiendas = cargar_datos_tiendas()
dir_activos = r"..\scripts\data\raw\maestro\8. Maestro de Reportes Agosto 2025.xlsb"
activos = cargar_datos_activos(dir_activos)

# Intercambios sugeridos
intercambios = intercambios_sugeridos(georref)

# Aplicar filtros desde el sidebar
filtros = sidebar_filters(georref)
datos_filtrados = aplicar_filtros(georref, *filtros)

# Cálculos
total_colaboradores = datos_filtrados.shape[0]
geolocalizados = total_colaboradores - datos_filtrados['distancia_actual_km'].isna().sum()
reubicaciones = datos_filtrados[datos_filtrados['mejora_%'] > 0].shape[0]
porcentaje_reubicaciones = (reubicaciones / geolocalizados) * 100 if geolocalizados > 0 else 0
distancia_promedio = datos_filtrados['distancia_actual_km'].mean()
tiempo_promedio = datos_filtrados['tiempo_actual_min'].mean()

# Mapa
datos_filtrados["tooltip_text"] = (
    "Colaborador: " + datos_filtrados['NOMBRE COMPLETO'].astype(str) +
    "\nID: " + datos_filtrados['NUMERO DE IDENTIFICACION'].astype(str) +
    "\n" + datos_filtrados['DESCRIPCION PUESTO'] +
    "\n" + datos_filtrados['LUGAR DE TRABAJO'] +
    "\nDistancia: " + datos_filtrados['distancia_actual_km'].astype(str) + " km" +
    "\nTiempo: " + datos_filtrados['tiempo_actual_min'].astype(str) + " min" +
    "\n" + datos_filtrados['079_DIRECCION']
)

# Layout
col_mapa, col_info = st.columns([1, 1.4])

with col_mapa:
    cantidad = len(datos_filtrados)
    zoom = max(6, min(10, 15 - math.log(cantidad + 1)))

    if len(datos_filtrados) != len(georref):
        centro_lat = datos_filtrados["latitud"].mean()
        centro_lon = datos_filtrados["longitud"].mean()
        view_state = pdk.ViewState(latitude=centro_lat, longitude=centro_lon, zoom=zoom)
    else:
        centro_lat, centro_lon = 4.60971, -74.08175
        view_state = pdk.ViewState(latitude=centro_lat, longitude=centro_lon, zoom=8)

    layers = [
        crear_capa_tiendas(datos_filtrados),
        crear_capa_colaboradores(datos_filtrados)
    ]

    if len(datos_filtrados) < len(georref):
        switch_state_arcos = st.toggle("Mostrar conexiones")
        if switch_state_arcos:
            layers.append(crear_arcos(datos_filtrados))
            
    if len(datos_filtrados) == 1:
        if not pd.isna(datos_filtrados['mejora_%'].iloc[0]):
            layers.append(crear_capa_tiendas_sugeridas(datos_filtrados, tiendas))
            if switch_state_arcos:
                layers.append(crear_arcos_tienda_sugerida(datos_filtrados, tiendas))

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{tooltip_text}"},
            map_style="mapbox://styles/mapbox/light-v9"
        ),
        use_container_width=True,
        height=475
    )

def inf_colaborador():
    if not pd.isna(datos_filtrados.iloc[0]['mejora_%']):
        colaborador_tienda_sugerida(datos_filtrados)
    else:
        mostrar_colaborador(datos_filtrados)

with col_info:
    if len(datos_filtrados) != 1:
        mostrar_metricas(total_colaboradores, tiempo_promedio, distancia_promedio,reubicaciones, geolocalizados)
        mostra_grafica(porcentaje_reubicaciones)
    else:
        if st.button("Mostrar intercambios sugeridos"):
            intercambio(intercambios, datos_filtrados.iloc[0]['NUMERO DE IDENTIFICACION'])
            if st.button("Volver"):
                inf_colaborador()
        else:
            inf_colaborador()
            
switch_state = st.toggle("Mostrar posibles reubicación")
if switch_state:
    mostrar_tabla(datos_filtrados[datos_filtrados['mejora_%'] > 0])
else:
    mostrar_tabla(datos_filtrados)