import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os
import urllib.request
import ssl
import sys

# ==============================================================================
# 0. PARCHE MAESTRO DE COMPATIBILIDAD (SKLEARN 1.5+ vs MODELO ANTIGUO)
# ==============================================================================
import sklearn.compose._column_transformer
import sklearn.impute._base

# 1. Parche para ColumnTransformer (Error _RemainderColsList)
class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
if 'sklearn.compose._column_transformer' in sys.modules:
    sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

# 2. Parche para SimpleImputer (Error _fill_dtype)
def get_fill_dtype(self):
    return getattr(self, "dtype", np.float64)

# Inyectamos la propiedad para que el modelo encuentre el atributo faltante
sklearn.impute._base.SimpleImputer._fill_dtype = property(get_fill_dtype)

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Predicción de Licitaciones",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CARGA DEL CEREBRO (GOOGLE DRIVE + SSL BYPASS)
# ==============================================================================
MAP_CONTRATO = {'W': 'Obras', 'U': 'Suministros', 'S': 'Servicios'}
MAP_PROCEDIMIENTO = {'OPE': 'Abierto', 'RES': 'Restringido', 'NEG': 'Negociado', 'COMP': 'Competitivo', 'OTH': 'Otro'}
MAP_CRITERIO = {'L': 'Precio más bajo', 'M': 'Mejor Relación (MEAT)', 'O': 'Mixto'}
MAP_ENTIDAD = {'1': 'Gobierno Central', '3': 'Autoridades Locales', '6': 'Organismos Públicos', '8': 'Otras', 'Z': 'No especificado'}
MAP_ACTIVIDAD = {
    'Health': 'Salud', 'Education': 'Educación', 'Defence': 'Defensa',
    'General public services': 'Servicios Públicos', 'Public order and safety': 'Seguridad',
    'Environment': 'Medio Ambiente', 'Economic and financial affairs': 'Economía',
    'Housing and community amenities': 'Vivienda', 'Social protection': 'Prot. Social',
    'Recreation, culture and religion': 'Cultura', 'Other': 'Otra'
}
MAP_PAIS = {'ES': 'España', 'FR': 'Francia', 'DE': 'Alemania', 'PL': 'Polonia', 'IT': 'Italia', 'PT': 'Portugal', 'NL': 'Países Bajos', 'BE': 'Bélgica'}

@st.cache_resource
def cargar_cerebro():
    ID_ARCHIVO_DRIVE = "1jOCGQTRZfNNoF1kGHD_S6OAxgUkLmC6c" 
    URL_DESCARGA = f"https://drive.google.com/uc?export=download&id={ID_ARCHIVO_DRIVE}"
    NOMBRE_LOCAL = "datos_tesis.joblib"
    
    try:
        if not os.path.exists(NOMBRE_LOCAL):
            with st.spinner('Descargando base de datos desde Google Drive...'):
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(URL_DESCARGA, context=context) as response, open(NOMBRE_LOCAL, 'wb') as out_file:
                    out_file.write(response.read())
        
        return joblib.load(NOMBRE_LOCAL)
    except Exception as e:
        st.error(f"🚨 Error crítico de carga: {e}")
        return None

sistema = cargar_cerebro()

if sistema is None:
    st.stop()

modelo = sistema['modelo_entrenado']
ref_participacion = sistema['ref_participacion']
ref_promedio_precio = sistema.get('ref_promedio_sector', sistema.get('ref_promedio_precio', {}))
ref_total_licitaciones = sistema.get('ref_total_licitaciones', sistema.get('ref_competencia_cpv', {}))
ref_promedio_competidores = sistema.get('ref_promedio_competidores', {})

# ==============================================================================
# 3. INTERFAZ GRÁFICA
# ==============================================================================
st.title("🏛️ Sistema de Viabilidad de Licitaciones")
st.markdown("**Análisis inteligente para PYMES en el mercado europeo**")
st.markdown("---")

if 'analisis_realizado' not in st.session_state:
    st.session_state['analisis_realizado'] = False

def resetear_analisis():
    st.session_state['analisis_realizado'] = False
    if 'resultado_base' in st.session_state: del st.session_state['resultado_base']

col_panel, col_result = st.columns([1, 1.5], gap="large")

with col_panel:
    st.subheader("1. Datos del Proyecto")
    valor_euro = st.number_input("Valor de tu Oferta (€)", min_value=0.0, value=150000.0, step=5000.0, on_change=resetear_analisis)
    num_ofertas = st.number_input("Competencia Estimada (Nº Empresas)", min_value=1, value=3, on_change=resetear_analisis)

    st.markdown("##### 📋 Detalles Técnicos")
    cpv_code = st.text_input("Código CPV", value="45000000", on_change=resetear_analisis)
    pais = st.selectbox("País", options=list(MAP_PAIS.keys()), format_func=lambda x: f"{MAP_PAIS.get(x)}", on_change=resetear_analisis)
    tipo_contrato = st.selectbox("Tipo Contrato", options=['W', 'U', 'S'], format_func=lambda x: MAP_CONTRATO.get(x), on_change=resetear_analisis)
    tipo_proc = st.selectbox("Procedimiento", options=['OPE', 'RES', 'NEG', 'COMP'], format_func=lambda x: MAP_PROCEDIMIENTO.get(x), on_change=resetear_analisis)
    criterio = st.selectbox("Criterio", options=['L', 'M', 'O'], format_func=lambda x: MAP_CRITERIO.get(x), on_change=resetear_analisis)
    tipo_entidad = st.selectbox("Entidad", options=['1', '3', '6', '8', 'Z'], format_func=lambda x: MAP_ENTIDAD.get(x), on_change=resetear_analisis)
    actividad = st.selectbox("Actividad", options=list(MAP_ACTIVIDAD.keys()), format_func=lambda x: MAP_ACTIVIDAD.get(x), on_change=resetear_analisis)

    empresa = st.text_input("Nombre del Licitante", placeholder="Ej: Mi Empresa S.A.", on_change=resetear_analisis)

    st.markdown("---")
    if st.button("🚀 Calcular Viabilidad", type="primary", use_container_width=True):
        st.session_state['analisis_realizado'] = True

# ==============================================================================
# 4. MOTOR DE CÁLCULO Y RESULTADOS
# ==============================================================================
if st.session_state['analisis_realizado']:
    with col_result:
        st.subheader("2. Resultados del Análisis")
        
        if 'resultado_base' not in st.session_state:
            with st.spinner('Procesando predicción...'):
                cpv_input = cpv_code.strip()
                promedio_sector = ref_promedio_precio.get(cpv_input, ref_promedio_precio.get(int(cpv_input) if cpv_input.isdigit() else None, valor_euro))
                historia = ref_participacion.get(empresa, 0)
                ratio_valor = valor_euro / (promedio_sector if promedio_sector != 0 else 1)
                comp_total = ref_total_licitaciones.get(cpv_input, 10)

                input_df = pd.DataFrame({
                    'Valor_Estimado_EUR': [valor_euro], 'Num_Ofertas_Recibidas': [num_ofertas],
                    'Participacion_Historica_Empresa': [historia], 'Competencia_Sector_CPV': [comp_total],
                    'Ratio_Valor_Sector': [ratio_valor], 'Codigo_CPV_Sector': [cpv_code], 'ISO_COUNTRY_CODE': [pais],
                    'TYPE_OF_CONTRACT': [tipo_contrato], 'Tipo_Procedimiento': [tipo_proc],
                    'MAIN_ACTIVITY': [actividad], 'CRIT_CODE': [criterio], 'CAE_TYPE': [tipo_entidad]
                })
                
                # Ejecución del modelo con parches activos
                prob_ml = modelo.predict_proba(input_df)[0][1]

                # Reglas de Tesis
                penalizacion = 0.0
                msgs = []
                if historia == 0: penalizacion += 0.125; msgs.append("📉 Sin historial: -12.5%")
                if num_ofertas >= 3: penalizacion += 0.20; msgs.append("👥 Competencia alta: -20%")
                
                prob_final = max(0.01, min(0.99, prob_ml - penalizacion))
                st.session_state['resultado_base'] = prob_final
                st.session_state['mensajes_base'] = msgs

        # Visualización Final
        prob_base = st.session_state['resultado_base']
        if prob_base > 0.5: st.success(f"### PROBABILIDAD DE ÉXITO: {prob_base:.2%}")
        else: st.error(f"### PROBABILIDAD DE ÉXITO: {prob_base:.2%}")
        st.progress(prob_base)
        
        for m in st.session_state.get('mensajes_base', []):
            st.caption(m)

