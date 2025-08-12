# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import plotly.express as px
from datetime import datetime
from pandas.tseries.offsets import DateOffset

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title = 'Predicción de Renta por m2', layout = 'wide')

# --- AUTENTICACIÓN SIMPLE ---
PASSWORD = 'ModeloConquer!'
password_input = st.sidebar.text_input('Ingresa la clave', type = 'password')
if password_input != PASSWORD:
    st.warning('🔒 Ingresa la clave correcta para acceder a la aplicación.')
    st.stop()

# --- CARGA DE MODELO Y PIPELINE ---
@st.cache_resource
def load_models():
    pipeline = joblib.load('pipeline_modelo.pkl')
    modelo = joblib.load('modelo_final_xgboost.pkl')
    return pipeline, modelo

pipeline, modelo = load_models()

# --- CARGA DE DATASET MAESTRO ---
@st.cache_data
def load_dataset_maestro():
    df = pd.read_excel('dataset_maestro.xlsx', sheet_name = 'dataset_maestro')
    df.drop(columns = ['mapeo.CONSTRUCCION'], errors = 'ignore', inplace = True)
    return df

df_maestro = load_dataset_maestro()

# --- FORMULARIO / ARCHIVO DE ENTRADA ---
st.title('📈 Predicción de Renta por Metro Cuadrado')

st.markdown('Carga un archivo con los siguientes campos, o ingrésalos manualmente:')
st.code("PLAZA, LOCAL, NOMBRE, GIRO, SUPERFICIE, MXN_POR_M2, FECHA_INICIO, FECHA_FIN, mapeo.UBICACION")

uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type = ['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file)
else:
    st.subheader('Captura manual (opcional)')
    with st.form('form_manual'):
        col1, col2, col3 = st.columns(3)
        with col1:
            plaza = st.selectbox('PLAZA', sorted(df_maestro['PLAZA'].dropna().unique()))
            giro = st.selectbox('GIRO', sorted(df_maestro['GIRO'].dropna().unique()))
        with col2:
            local = st.text_input('LOCAL (ej. A101)', value = 'A101').upper()
            superficie = st.number_input('SUPERFICIE', min_value = 1.0, step = 1.0)
        with col3:
            nombre = st.text_input('NOMBRE', value = 'NUEVO LOCAL').upper()
            ubicacion = st.selectbox('mapeo.UBICACION', sorted(df_maestro['mapeo.UBICACION'].dropna().unique()))

        col4, col5 = st.columns(2)
        with col4:
            fecha_inicio = st.date_input('FECHA_INICIO', value = datetime.today())
        with col5:
            fecha_fin = st.date_input('FECHA_FIN', value = datetime.today())

        mxn_m2 = st.number_input('MXN_POR_M2 (opcional)', min_value = 0.0, step = 10.0)
        enviar = st.form_submit_button('Agregar entrada')

    if enviar:
        df_input = pd.DataFrame([{
            'PLAZA': plaza, 'LOCAL': local, 'NOMBRE': nombre,
            'GIRO': giro, 'SUPERFICIE': superficie,
            'MXN_POR_M2': mxn_m2 if mxn_m2 > 0 else np.nan,
            'FECHA_INICIO': pd.to_datetime(fecha_inicio),
            'FECHA_FIN': pd.to_datetime(fecha_fin),
            'mapeo.UBICACION': ubicacion
        }])

# --- PROCESAMIENTO SI HAY DATOS DE ENTRADA ---
if 'df_input' in locals():
    st.success('✅ Datos recibidos correctamente.')

    # --- CRUCE CON DATASET MAESTRO ---
    df_joined = df_input.merge(
        df_maestro.drop(columns = ['MXN_POR_M2', 'FECHA_INICIO', 'FECHA_FIN', 'PLAZA', 'LOCAL', 'NOMBRE', 'GIRO', 'SUPERFICIE']),
        on = ['mapeo.UBICACION', 'PLAZA'],
        how = 'left'
    )

    # Si no hubo match por PLAZA + UBICACION, intentar solo por UBICACION
    if df_joined.isnull().any().any():
        df_joined = df_input.merge(
            df_maestro.drop(columns = ['MXN_POR_M2', 'FECHA_INICIO', 'FECHA_FIN', 'PLAZA', 'LOCAL', 'NOMBRE', 'GIRO', 'SUPERFICIE']),
            on = ['mapeo.UBICACION'],
            how = 'left'
        )

    df = df_joined.copy()
    df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO'], errors = 'coerce')
    df['FECHA_FIN'] = pd.to_datetime(df['FECHA_FIN'], errors = 'coerce')
    df['GIRO'] = df['GIRO'].fillna('SIN CLASIFICAR')

    # Variables derivadas
    df['DURACION_MESES'] = ((df['FECHA_FIN'] - df['FECHA_INICIO']).dt.days / 30.44).round(1)
    df['ESTA_VIGENTE'] = (df['FECHA_FIN'] >= pd.Timestamp.today()).astype(int)
    df['ANTIGÜEDAD_MESES'] = ((pd.Timestamp.today() - df['FECHA_INICIO']).dt.days / 30.44).round(1)
    df['ID_LOCAL'] = df['PLAZA'].astype(str) + ' - ' + df['LOCAL'].astype(str) + ' - ' + df['mapeo.UBICACION'].astype(str)
    df['ES_KIOSKO'] = (df['GIRO'].str.upper().str.strip() == 'KIOSKOS').astype(int)
    df['VINTAGE_INICIO'] = df['FECHA_INICIO'].dt.strftime('%Y%m').astype(int)
    df['VINTAGE_FIN'] = df['FECHA_FIN'].dt.strftime('%Y%m').astype(int)

    # Clasificación por tamaño
    def clasificar_tamano(m2):
        if m2 < 20:
            return 'KIOSKO'
        elif m2 <= 150:
            return 'SMALL'
        elif m2 <= 500:
            return 'MEDIUM'
        else:
            return 'LARGE'

    df['TAMANO_LOCAL'] = df['SUPERFICIE'].apply(clasificar_tamano)

    # --- PREDICCIÓN ---
    X_proc = pipeline.transform(df)
    y_pred_log = modelo.predict(X_proc)
    df['PREDICCIÓN_LOG'] = y_pred_log
    df['PREDICCIÓN_MXN_POR_M2'] = np.expm1(y_pred_log)

    # --- VISUALIZACIÓN DE RESULTADOS ---
    st.subheader('📋 Resultados de la Predicción')

    with st.expander('🔍 Ver tabla completa'):
        st.dataframe(df.style.format({'PREDICCIÓN_MXN_POR_M2': '{:.2f}'}), height = 400)

    # --- DESCARGA ---
    def convertir_csv(df):
        return df.to_csv(index = False).encode('utf-8')

    st.download_button(
        label = '⬇️ Descargar resultados como CSV',
        data = convertir_csv(df),
        file_name = 'predicciones_renta.csv',
        mime = 'text/csv'
    )

    # --- GRÁFICO ---
    if df['GIRO'].nunique() > 1:
        fig = px.scatter(
            df,
            x = 'SUPERFICIE',
            y = 'PREDICCIÓN_MXN_POR_M2',
            color = 'GIRO',
            hover_name = 'NOMBRE',
            size = 'SUPERFICIE',
            title = 'Predicción por tamaño y giro',
            labels = {'SUPERFICIE': 'm²', 'PREDICCIÓN_MXN_POR_M2': '$/m² Predicho'}
        )
    else:
        fig = px.scatter(
            df,
            x = 'SUPERFICIE',
            y = 'PREDICCIÓN_MXN_POR_M2',
            hover_name = 'NOMBRE',
            size = 'SUPERFICIE',
            title = 'Predicción por tamaño',
            labels = {'SUPERFICIE': 'm²', 'PREDICCIÓN_MXN_POR_M2': '$/m² Predicho'}
        )

    st.plotly_chart(fig, use_container_width = True)
