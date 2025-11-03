import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def mostra_grafica (porcentaje):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=porcentaje,
        number={'font': {'size': 48}},
        title={'text': "Posibles reubicaciones (%)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1a5276"},
            'steps': [
                {'range': [0, 50], 'color': "#7fb3d5"},
                {'range': [50, 80], 'color': "#5499c7"},
                {'range': [80, 100], 'color': "#2980b9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': porcentaje
            }
        }
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=10), height=250)
    st.plotly_chart(fig, use_container_width=True)


def mostrar_metricas(total, tiempo, promedio, reubicaciones, geolocalizados):
    col1, col2, col3 = st.columns([1,1,1])

    col1.metric("Total colaboradores", value=total, border= True)
    col2.metric("Distancia promedio", value=f"{promedio:.1f} Km", border= True)
    col3.metric("Tiempo promedio", value=f"{tiempo:.1f} Min", border= True)

    col1, col2 = st.columns([1,2])
    col1.metric("Reubicaciones sugeridas", value=reubicaciones, border= True)
    col2.markdown(f"Para este análisis se está teniendo en cuenta **{geolocalizados}** del total de colaboradores. La cantidad de colaboradores que se lograron geolocalizar.")



@st.cache_data
def mostrar_tabla(df):
    columnas = ['NUMERO DE IDENTIFICACION', 'NOMBRE COMPLETO', 'DESCRIPCION PUESTO',
                'LUGAR DE TRABAJO', 'Antigüedad', 'DESCRIPCION TIEMPO TEORICO', '079_DIRECCION', 'distancia_actual_km', 'tiempo_actual_min',
                'tienda_mas_cercana_diferente', 'distancia_sugerido_km', 'tiempo_sugerido_min',
                'mejora_%']

    df = df[columnas].copy()

    df = df.rename(columns={
        '079_DIRECCION': 'DIRECCIÓN',
        'distancia_actual_km': 'DISTANCIA (KM)',
        'tiempo_actual_min': 'TIEMPO (MIN)',
        'tienda_mas_cercana_diferente': 'TIENDA SUGERIDA',
        'distancia_sugerido_km': 'DISTANCIA SUGERIDA (KM)',
        'tiempo_sugerido_min': 'TIEMPO SUGERIDO (MIN)',
        'mejora_%': 'MEJORA (%)'
    })

    df['MEJORA (%)'] = df['MEJORA (%)']/100 

    df_estilizado = df.style \
        .format({
            'DISTANCIA (KM)': '{:.2f}',
            'TIEMPO (MIN)': '{:.0f}',
            'DISTANCIA SUGERIDA (KM)': '{:.2f}',
            'TIEMPO SUGERIDO (MIN)': '{:.0f}',
            'MEJORA (%)': '{:.1%}',
            'Antigüedad' : '{:.1f}'
        }) \
        .apply(lambda x: ['background-color: #d4edda' if v > 0 else '' for v in x], subset=['MEJORA (%)'])

    st.dataframe(df_estilizado, use_container_width=True)


def mostrar_colaborador (df):
    fila = df.iloc[0] 

    st.markdown(f"<p style='font-size:16px; font-weight:bold;'>{fila['NOMBRE COMPLETO']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:14px;'>{fila['DESCRIPCION PUESTO']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:12px;'>{fila['079_DIRECCION']}</p>", unsafe_allow_html=True)

    st.markdown('Lugar de trabajo actual:')
    st.markdown(f"<p style='font-size:16px; font-weight:bold;'>{fila['LUGAR DE TRABAJO']}</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="border:1px solid #ccc; border-radius:5px; padding:8px; text-align:center;">
            <div style="font-size:12px; color:gray;">Distancia</div>
            <div style="font-size:16px; font-weight:bold;">{fila['distancia_actual_km']} Km</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="border:1px solid #ccc; border-radius:5px; padding:8px; text-align:center;">
            <div style="font-size:12px; color:gray;">Tiempo</div>
            <div style="font-size:16px; font-weight:bold;">{fila['tiempo_actual_min']} Min</div>
        </div>
        """, unsafe_allow_html=True)

def colaborador_tienda_sugerida(df):
    fila = df.iloc[0] 

    st.markdown(f"<p style='font-size:16px; font-weight:bold;'>{fila['NOMBRE COMPLETO']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:14px;'>{fila['DESCRIPCION PUESTO']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:12px;'>{fila['079_DIRECCION']}</p>", unsafe_allow_html=True)

    st.markdown('Lugar de trabajo actual:')
    st.markdown(f"<p style='font-size:16px; font-weight:bold;'>{fila['LUGAR DE TRABAJO']}</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="border:1px solid #ccc; border-radius:5px; padding:8px; text-align:center;">
            <div style="font-size:12px; color:gray;">Distancia</div>
            <div style="font-size:16px; font-weight:bold;">{fila['distancia_actual_km']} Km</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="border:1px solid #ccc; border-radius:5px; padding:8px; text-align:center;">
            <div style="font-size:12px; color:gray;">Tiempo</div>
            <div style="font-size:16px; font-weight:bold;">{fila['tiempo_actual_min']} Min</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('Lugar de trabajo sugerido:')
    st.markdown(f"<p style='font-size:16px; font-weight:bold;'>{fila['tienda_mas_cercana_diferente']}</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="border:1px solid #FFA500; background-color:#FFF5E5; border-radius:5px; padding:8px; text-align:center;">
            <div style="font-size:12px; color:gray;">Distancia</div>
            <div style="font-size:16px; font-weight:bold;">{fila['distancia_sugerido_km']} Km</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="border:1px solid #FFA500; background-color:#FFF5E5; border-radius:5px; padding:8px; text-align:center;">
            <div style="font-size:12px; color:gray;">Tiempo</div>
            <div style="font-size:16px; font-weight:bold;">{fila['tiempo_sugerido_min']} Min</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="border:1px solid #2ebf11; background-color:#b9f7ac; border-radius:5px; padding:8px; width:200px; text-align:center;">
            <div style="font-size:12px; color:#2ebf11;">Mejora</div>
            <div style="font-size:16px; font-weight:bold; color:#2ebf11;">{fila['mejora_%']}%</div>
        </div>
        """, unsafe_allow_html=True)



def intercambio(df, id):
    df = df[df['ID_Empleado_1'] == id]
    st.dataframe(df, use_container_width=True)
