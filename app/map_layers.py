import pydeck as pdk
import pandas as pd

def crear_capa_tiendas(df):
    datos = df[["LUGAR DE TRABAJO", "latitud", "longitud"]].drop_duplicates()
    datos["tooltip_text"] = datos["LUGAR DE TRABAJO"]
    return pdk.Layer(
        "ScatterplotLayer",
        data=datos,
        get_position=["longitud", "latitud"],
        get_radius=130,
        radius_min_pixels=5,
        radius_max_pixels=35,
        get_fill_color='[0, 112, 255, 200]',
        pickable=True
    )

def crear_capa_colaboradores(df):
    return pdk.Layer(
        "ScatterplotLayer",
        data=df[~(df['distancia_actual_km'].isna())],
        get_position=["lng", "lat"],
        get_radius=60,
        radius_min_pixels=2,
        radius_max_pixels=15,
        get_fill_color='[120, 120, 120, 100]',
        pickable=True
    )

def crear_arcos(df):
    return pdk.Layer(
        "ArcLayer",
        data=df[~(df['distancia_actual_km'].isna())],
        get_source_position=["longitud", "latitud"],
        get_target_position=["lng", "lat"],
        get_width="S000 * 10",
        get_source_color=[0, 0, 0, 20],
        get_target_color=[0, 0, 0, 50],
        get_tilt=15,
        pickable=True,
        auto_highlight=True
    )

def crear_capa_tiendas_sugeridas(df, tiendas):
    lugares_sugeridos = df[["tienda_mas_cercana_diferente"]].dropna().drop_duplicates()
    lugares_sugeridos = lugares_sugeridos.rename(columns={"tienda_mas_cercana_diferente": "LUGAR DE TRABAJO"})
    datos = lugares_sugeridos.merge(
        tiendas[["LUGAR DE TRABAJO", "latitud", "longitud"]],
        on="LUGAR DE TRABAJO",
        how="left"
    )
    datos["tooltip_text"] = datos["LUGAR DE TRABAJO"]
    return pdk.Layer(
        "ScatterplotLayer",
        data=datos,
        get_position=["longitud", "latitud"],
        get_radius=130,
        radius_min_pixels=5,
        radius_max_pixels=35,
        get_fill_color='[255, 140, 0, 200]', 
        pickable=True
    )

def crear_arcos_tienda_sugerida(df, tiendas):
    df = df[['lat', 'lng', 'tienda_mas_cercana_diferente']].rename(columns={"tienda_mas_cercana_diferente": "LUGAR DE TRABAJO"})
    datos = pd.merge(
        df,
        tiendas[["LUGAR DE TRABAJO", "latitud", "longitud"]],
        on="LUGAR DE TRABAJO",
        how="left"
    )

    datos["tooltip_text"] = datos["LUGAR DE TRABAJO"]     
    
    return pdk.Layer(
        "ArcLayer",
        data=datos[["latitud", "longitud", "lat", "lng", "tooltip_text"]],
        get_target_position=["longitud", "latitud"],
        get_source_position=["lng", "lat"],
        get_source_color=[30, 144, 255, 80],  
        get_target_color=[255, 140, 0, 120], 
        get_tilt=15,
        pickable=True,
        auto_highlight=True
    )
