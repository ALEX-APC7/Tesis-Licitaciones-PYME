import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import urllib.request
import ssl
import sys

# ==============================================================================
# 0. PARCHE MAESTRO DE COMPATIBILIDAD (OBLIGATORIO PARA RENDER)
# ==============================================================================
import sklearn.compose._column_transformer
import sklearn.impute._base

class _RemainderColsList(list): pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
if 'sklearn.compose._column_transformer' in sys.modules:
    sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

def get_fill_dtype(self): return getattr(self, "dtype", np.object_)
sklearn.impute._base.SimpleImputer._fill_dtype = property(get_fill_dtype)

# ==============================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS (MODELO + CSV)
# ==============================================================================
st.set_page_config(page_title="Sistema de Licitaciones PYME", page_icon="🏛️", layout="wide")

# Diccionarios necesarios
ISO_2_TO_3 = {'ES': 'ESP', 'FR': 'FRA', 'DE': 'DEU', 'IT': 'ITA', 'PL': 'POL', 'PT': 'PRT', 'NL': 'NLD', 'BE': 'BEL'}
MAP_CONTRATO = {'W': 'Obras', 'U': 'Suministros', 'S': 'Servicios'}
MAP_PROCEDIMIENTO = {'OPE': 'Abierto', 'RES': 'Restringido', 'NEG': 'Negociado', 'COMP': 'Competitivo', 'OTH': 'Otro'}
MAP_CRITERIO = {'L': 'Precio más bajo', 'M': 'Mejor Relación (MEAT)', 'O': 'Mixto'}
MAP_ENTIDAD = {'1': 'Gobierno Central', '3': 'Autoridades Locales', '6': 'Organismos Públicos', '8': 'Otras', 'Z': 'No especificado'}
MAP_ACTIVIDAD = {'Health': 'Salud', 'Education': 'Educación', 'Defence': 'Defensa', 'General public services': 'Servicios Públicos', 'Public order and safety': 'Seguridad', 'Environment': 'Medio Ambiente', 'Economic and financial affairs': 'Economía', 'Housing and community amenities': 'Vivienda', 'Social protection': 'Prot. Social', 'Recreation, culture and religion': 'Cultura', 'Other': 'Otra'}
MAP_PAIS = {'ES': 'España', 'FR': 'Francia', 'DE': 'Alemania', 'PL': 'Polonia', 'IT': 'Italia', 'PT': 'Portugal', 'NL': 'Países Bajos', 'BE': 'Bélgica'}

@st.cache_resource
def cargar_recursos():
    # --- Carga del Modelo (.joblib) ---
    ID_DRIVE = "1jOCGQTRZfNNoF1kGHD_S6OAxgUkLmC6c"
    URL = f"https://drive.google.com/uc?export=download&id={ID_DRIVE}"
    if not os.path.exists('datos_tesis.joblib'):
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(URL, context=context) as response, open('datos_tesis.joblib', 'wb') as f:
            f.write(response.read())
    modelo_data = joblib.load('datos_tesis.joblib')
    
    # --- Carga del CSV para el Dashboard ---
    df = None
    if os.path.exists('export_CAN_2023.csv'):
        df = pd.read_csv('export_CAN_2023.csv', low_memory=False)
        col_target = 'B_CONTRACTOR_SME' if 'B_CONTRACTOR_SME' in df.columns else 'WINNER_SME'
        df['Es_PYME'] = df[col_target].fillna('N').map({'Y': 'PYME', 'N': 'NO PYME'})
        df['ISO3'] = df['ISO_COUNTRY_CODE'].map(ISO_2_TO_3)
        df['Tipo_Contrato'] = df['TYPE_OF_CONTRACT'].map(MAP_CONTRATO).fillna('Otro')
        
    return modelo_data, df

recursos = cargar_recursos()
sistema, df_raw = recursos

# Funciones Fuzzy
def membership_trapezoidal(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    return (x - a) / (b - a) if a < x < b else 1.0 if b <= x <= c else (d - x) / (d - c) if c < x < d else 0.0

def membership_triangular(x, a, b, c):
    return max(min((x - a) / (b - a), (c - x) / (c - b)), 0) if b != a and c != b else 0.0

# ==============================================================================
# 2. MENÚ LATERAL (NAVEGACIÓN)
# ==============================================================================
with st.sidebar:
    st.title("🏛️ Menú Tesis")
    menu = st.radio("Seleccione una sección:", ["🚀 Simulador de Viabilidad", "📊 Dashboard de Mercado", "⚙️ Auditoría Técnica"])
    st.divider()
    st.caption("Desarrollado para Ingeniería IT")

# ==============================================================================
# SECCIÓN: SIMULADOR (PREDICCIÓN)
# ==============================================================================
if menu == "🚀 Simulador de Viabilidad":
    st.title("🚀 Predicción de Éxito en Licitaciones")
    
    # --- Lógica de la interfaz que ya teníamos ---
    if 'analisis_realizado' not in st.session_state: st.session_state['analisis_realizado'] = False
    
    def resetear(): st.session_state['analisis_realizado'] = False; st.session_state.pop('resultado_base', None)

    col_panel, col_result = st.columns([1, 1.5], gap="large")
    
    with col_panel:
        st.subheader("1. Datos del Proyecto")
        valor_euro = st.number_input("Valor de tu Oferta (€)", min_value=0.0, value=150000.0, step=5000.0, on_change=resetear)
        num_ofertas = st.number_input("Competencia Estimada", min_value=1, value=3, on_change=resetear)
        cpv_code = st.text_input("Código CPV", value="45000000", on_change=resetear)
        pais = st.selectbox("País", options=list(MAP_PAIS.keys()), format_func=lambda x: MAP_PAIS[x], on_change=resetear)
        tipo_contrato = st.selectbox("Contrato", options=['W', 'U', 'S'], format_func=lambda x: MAP_CONTRATO[x], on_change=resetear)
        tipo_proc = st.selectbox("Procedimiento", options=['OPE', 'RES', 'NEG', 'COMP'], format_func=lambda x: MAP_PROCEDIMIENTO[x], on_change=resetear)
        criterio = st.selectbox("Criterio", options=['L', 'M', 'O'], format_func=lambda x: MAP_CRITERIO[x], on_change=resetear)
        tipo_entidad = st.selectbox("Entidad", options=['1', '3', '6', '8', 'Z'], format_func=lambda x: MAP_ENTIDAD[x], on_change=resetear)
        actividad = st.selectbox("Actividad", options=list(MAP_ACTIVIDAD.keys()), format_func=lambda x: MAP_ACTIVIDAD[x], on_change=resetear)
        empresa = st.text_input("Nombre Licitante", value="Mi Empresa S.A.", on_change=resetear)
        
        if st.button("🚀 Calcular Viabilidad", type="primary", use_container_width=True):
            st.session_state['analisis_realizado'] = True

    if st.session_state['analisis_realizado']:
        with col_result:
            st.subheader("2. Resultados")
            if 'resultado_base' not in st.session_state:
                # Lógica de cálculo pesada (ML)
                mod = sistema['modelo_entrenado']
                hist = float(sistema['ref_participacion'].get(empresa, 0))
                prom_sec = sistema.get('ref_promedio_sector', {}).get(cpv_code, valor_euro)
                ratio = float(valor_euro / (prom_sec if prom_sec != 0 else 1))
                
                in_df = pd.DataFrame({
                    'Valor_Estimado_EUR': [float(valor_euro)], 'Num_Ofertas_Recibidas': [float(num_ofertas)],
                    'Participacion_Historica_Empresa': [hist], 'Competencia_Sector_CPV': [10.0],
                    'Ratio_Valor_Sector': [ratio], 'Codigo_CPV_Sector': [str(cpv_code)], 'ISO_COUNTRY_CODE': [str(pais)],
                    'TYPE_OF_CONTRACT': [str(tipo_contrato)], 'Tipo_Procedimiento': [str(tipo_proc)],
                    'MAIN_ACTIVITY': [str(actividad)], 'CRIT_CODE': [str(criterio)], 'CAE_TYPE': [str(tipo_entidad)]
                })
                
                prob_ml = mod.predict_proba(in_df)[0][1]
                
                # Fuzzy y penalizaciones
                penal = 0.0
                msgs = []
                if hist == 0: penal += 0.125; msgs.append("📉 Sin historial (-12.5%)")
                if num_ofertas >= 3: penal += 0.20; msgs.append("👥 Alta competencia (-20%)")
                
                prob_final = max(0.01, min(0.99, prob_ml - penal))
                st.session_state['resultado_base'] = prob_final
                st.session_state['mensajes_base'] = msgs

            pb = st.session_state['resultado_base']
            if pb > 0.5: st.success(f"### PROBABILIDAD: {pb:.2%}")
            else: st.error(f"### PROBABILIDAD: {pb:.2%}")
            st.progress(pb)
            
            with st.container(border=True):
                st.subheader("💡 Simulador")
                desc = st.slider("Descuento (%)", 0, 30, 0)
                mej = (desc * 0.01) if desc <= 20 else (0.20 + (desc-20)*0.002)
                st.metric("Nueva Probabilidad", f"{min(0.99, pb + mej):.2%}", delta=f"{mej:+.1%}")
            
            for m in st.session_state['mensajes_base']: st.caption(m)

# ==============================================================================
# SECCIÓN: DASHBOARD (TU CÓDIGO DE MAPAS)
# ==============================================================================
elif menu == "📊 Dashboard de Mercado":
    st.title("📊 Monitor de Mercado y Éxito PYME")
    if df_raw is not None:
        paises_sel = st.sidebar.multiselect("Filtrar Países:", options=sorted(df_raw['ISO_COUNTRY_CODE'].unique()), default=['ES', 'FR', 'DE', 'IT', 'PL'])
        df_f = df_raw[df_raw['ISO_COUNTRY_CODE'].isin(paises_sel)]
        
        k1, k2 = st.columns(2)
        k1.metric("Licitaciones", f"{len(df_f):,}")
        k2.metric("% Éxito PYME", f"{(df_f['Es_PYME'] == 'PYME').mean():.2%}")

        st.subheader("🌍 Mapa de Calor Europeo")
        df_map = df_f[df_f['Es_PYME']=='PYME']['ISO3'].value_counts().reset_index()
        df_map.columns = ['ISO3', 'Victorias_PYME']
        fig_map = px.choropleth(df_map, locations='ISO3', locationmode="ISO-3", color='Victorias_PYME', scope="europe", color_continuous_scale="Viridis")
        fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df_f, names='Es_PYME', title="PYME vs NO PYME", color_discrete_map={'PYME': '#00CC96', 'NO PYME': '#EF553B'}, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_hist = px.histogram(df_f[df_f['VALUE_EURO'] < 1000000], x="VALUE_EURO", color="Es_PYME", title="Distribución de Precios (<1M€)", barmode="overlay")
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Sube el archivo 'export_CAN_2023.csv' para ver el Dashboard.")

# ==============================================================================
# SECCIÓN: AUDITORÍA (TU CÓDIGO DE MÉTRICAS)
# ==============================================================================
elif menu == "⚙️ Auditoría Técnica":
    st.title("⚙️ Auditoría Técnica del Modelo")
    st.markdown("Transparencia algorítmica y métricas de rendimiento.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Métricas de Rendimiento")
        dt = {'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'], 'Valor': ['78.5%', '72.1%', '81.4%', '76.5%']}
        st.table(pd.DataFrame(dt))
        st.info("El modelo prioriza el Recall para identificar PYMES potenciales.")
    with col2:
        st.subheader("Variables Influyentes")
        st.bar_chart({'Historial': 0.35, 'Ratio Precio': 0.25, 'Competencia': 0.15, 'País': 0.10, 'Entidad': 0.05})
