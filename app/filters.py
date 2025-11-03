import streamlit as st
import re

from utils.columnas import contar_y_ultimo_elemento

dir_maestro = r"..\scripts\data\raw\maestro"
n_maestro, lista_maestro = contar_y_ultimo_elemento(dir_maestro)
ultimo_maestro = sorted(lista_maestro)[-1]

patron = r"\b([A-Za-záéíóúÁÉÍÓÚñÑ]+ \d{4})\b"
coincidencia = re.search(patron, ultimo_maestro)

def sidebar_filters(df):
    st.sidebar.markdown(f":gray[{coincidencia.group(1)}]")
    st.sidebar.title("Filtros")
    #id_colaborador = st.sidebar.text_input("Buscar por número de identificación")
    id_colaborador = str(st.sidebar.selectbox("Selecciona ID de colaborador", ["Todos"] + list(df['NUMERO DE IDENTIFICACION'].unique())))
    # ciudad
    ciudades = st.sidebar.multiselect("Ciudad", df["CIUDAD DONDE TRABAJA"].unique())
    if ciudades:
        df_filtrado = df[df["CIUDAD DONDE TRABAJA"].isin(ciudades)]
    else:
        df_filtrado = df
    tiendas = st.sidebar.multiselect("Lugar de trabajo", df_filtrado["LUGAR DE TRABAJO"].unique())
    cargo = st.sidebar.multiselect("Cargo", df["DESCRIPCION PUESTO"].unique())
    return id_colaborador, tiendas, ciudades, cargo

def aplicar_filtros(df, id_colaborador, tiendas, ciudades, cargo):
    datos = df.copy()
    if id_colaborador == 'Todos':
        pass
    else:
        datos = datos[datos['NUMERO DE IDENTIFICACION'].astype(str).str.contains(id_colaborador)]
    if tiendas:
        datos = datos[datos['LUGAR DE TRABAJO'].isin(tiendas)]
    if ciudades:
        datos = datos[datos['CIUDAD DONDE TRABAJA'].isin(ciudades)]
    if cargo:
        datos = datos[datos['DESCRIPCION PUESTO'].isin(cargo)]
    if datos.empty:
        #st.warning("No se encontraron resultados del filtro aplicado. ")
        return df
    return datos
