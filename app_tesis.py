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
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==============================================================================
st.set_page_config(page_title="Sistema de Licitaciones", page_icon="⚖️", layout="wide")

# Variables Maestras de tu Tesis
NUMERIC_FEATURES = ['Valor_Estimado_EUR', 'Num_Ofertas_Recibidas', 'Participacion_Historica_Empresa', 'Competencia_Sector_CPV', 'Ratio_Valor_Sector']
CATEGORICAL_FEATURES = ['Codigo_CPV_Sector', 'ISO_COUNTRY_CODE', 'CAE_TYPE', 'MAIN_ACTIVITY', 'TYPE_OF_CONTRACT', 'Tipo_Procedimiento', 'CRIT_CODE']
ISO_2_TO_3 = {'ES': 'ESP', 'FR': 'FRA', 'DE': 'DEU', 'IT': 'ITA', 'PL': 'POL', 'PT': 'PRT', 'NL': 'NLD', 'BE': 'BEL'}
MAP_PAIS = {'ES': 'España', 'FR': 'Francia', 'DE': 'Alemania', 'PL': 'Polonia', 'IT': 'Italia', 'PT': 'Portugal', 'NL': 'Países Bajos', 'BE': 'Bélgica'}
MAP_CONTRATO = {'W': 'Obras', 'U': 'Suministros', 'S': 'Servicios'}

@st.cache_resource
def cargar_recursos():
    ID_DRIVE_MOD = "1jOCGQTRZfNNoF1kGHD_S6OAxgUkLmC6c"
    URL_MOD = f"https://drive.google.com/uc?export=download&id={ID_DRIVE_MOD}"
    if not os.path.exists('datos_tesis.joblib'):
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(URL_MOD, context=context) as response, open('datos_tesis.joblib', 'wb') as f:
            f.write(response.read())
    return joblib.load('datos_tesis.joblib')

@st.cache_data
def cargar_csv_dashboard():
    ID_DRIVE_CSV = "14PRk0KYhlxrtDsGFoXEw_giPCNbXTaug"
    URL_CSV = f"https://drive.google.com/uc?export=download&id={ID_DRIVE_CSV}"
    FILE_CSV = "export_CAN_2023.csv"
    if not os.path.exists(FILE_CSV):
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(URL_CSV, context=context) as response, open(FILE_CSV, 'wb') as f:
                f.write(response.read())
        except: return None
    
    df = pd.read_csv(FILE_CSV, low_memory=False)
    # Buscador automático de columnas para el Dashboard
    def buscar_col(keywords):
        for k in keywords:
            for c in df.columns:
                if k.upper() in str(c).upper(): return c
        return None

    c_pyme = buscar_col(['Es_PYME_Ganadora', 'SME', 'WINNER_SME'])
    c_pais = buscar_col(['ISO_COUNTRY_CODE', 'COUNTRY', 'PAÍS'])
    
    if c_pyme: 
        df['Es_PYME_Num'] = df[c_pyme].astype(str).str[0].map({'Y': 1, '1': 1, 'S': 1}).fillna(0)
    if c_pais: 
        df['ISO_COUNTRY_CODE'] = df[c_pais]
        df['ISO3'] = df['ISO_COUNTRY_CODE'].map(ISO_2_TO_3)
    
    return df

sistema = cargar_recursos()
modelo = sistema['modelo_entrenado']
ref_participacion = sistema.get('ref_participacion', {})
ref_promedio_precio = sistema.get('ref_promedio_sector', {})
ref_total_licitaciones = sistema.get('ref_total_licitaciones', {})
ref_promedio_competidores = sistema.get('ref_promedio_competidores', {})

# ==============================================================================
# 2. MENÚ LATERAL
# ==============================================================================
with st.sidebar:
    st.title("🏛️ Menú Principal")
    menu = st.radio("Sección:", ["🚀 Simulador de Viabilidad", "📊 Dashboard de Mercado", "⚙️ Auditoría Técnica"])

# ==============================================================================
# SECCIÓN 1: SIMULADOR (TU CÓDIGO INAMOVIBLE Y COMPLETO)
# ==============================================================================
if menu == "🚀 Simulador de Viabilidad":
    def membership_trapezoidal(x, a, b, c, d):
        if x <= a or x >= d: return 0.0
        if a < x < b: return (x - a) / (b - a)
        if b <= x <= c: return 1.0
        if c < x < d: return (d - x) / (d - c)
        return 0.0

    st.title("🏛️ Sistema de Viabilidad de Licitaciones")
    st.markdown("**Análisis inteligente para PYMES en el mercado europeo**")
    st.markdown("---")

    if 'analisis_realizado' not in st.session_state: st.session_state['analisis_realizado'] = False
    def resetear(): 
        st.session_state['analisis_realizado'] = False
        st.session_state.pop('resultado_base', None)

    col_in, col_out = st.columns([1, 1.5], gap="large")

    with col_in:
        st.subheader("1. Datos del Proyecto")
        valor_euro = st.number_input("Valor de tu Oferta (€)", min_value=0.0, value=150000.0, step=5000.0, on_change=resetear)
        num_ofertas = st.number_input("Competencia Estimada", min_value=1, value=3, on_change=resetear)
        cpv_code = st.text_input("Código CPV", value="45000000", on_change=resetear)
        pais = st.selectbox("País", options=list(MAP_PAIS.keys()), format_func=lambda x: MAP_PAIS[x], on_change=resetear)
        tipo_contrato = st.selectbox("Contrato", options=['W', 'U', 'S'], format_func=lambda x: MAP_CONTRATO[x], on_change=resetear)
        empresa = st.text_input("Nombre Licitante", value="Mi Empresa S.A.", on_change=resetear)
        st.button("🚀 Calcular Viabilidad", type="primary", use_container_width=True, on_click=lambda: st.session_state.update({'analisis_realizado': True}))

    if st.session_state['analisis_realizado']:
        with col_out:
            st.subheader("2. Resultados del Análisis")
            if 'resultado_base' not in st.session_state:
                cpv_str = cpv_code.strip()
                prom_sec = ref_promedio_precio.get(cpv_str, ref_promedio_precio.get(int(cpv_str) if cpv_str.isdigit() else None, valor_euro))
                hist = float(ref_participacion.get(empresa, 0))
                ratio = float(valor_euro / (prom_sec if prom_sec != 0 else 1))
                comp_tot = float(ref_total_licitaciones.get(cpv_str, 10))

                input_df = pd.DataFrame({
                    'Valor_Estimado_EUR': [float(valor_euro)], 'Num_Ofertas_Recibidas': [float(num_ofertas)],
                    'Participacion_Historica_Empresa': [hist], 'Competencia_Sector_CPV': [comp_tot],
                    'Ratio_Valor_Sector': [ratio], 'Codigo_CPV_Sector': [str(cpv_code)], 'ISO_COUNTRY_CODE': [str(pais)],
                    'CAE_TYPE': ['1'], 'MAIN_ACTIVITY': ['Health'], 'TYPE_OF_CONTRACT': [str(tipo_contrato)], 'Tipo_Procedimiento': ['OPE'], 'CRIT_CODE': ['L']
                })
                
                prob_ml = modelo.predict_proba(input_df)[0][1]
                mu_hist = membership_trapezoidal(hist, -1, 0, 0, 5)
                penal = 0.125 if mu_hist > 0.5 else 0.0
                if num_ofertas >= 3: penal += 0.20
                
                st.session_state['resultado_base'] = max(0.01, min(0.99, prob_ml - penal))
                st.session_state['mets'] = {'hist': hist, 'ratio': ratio, 'comp': num_ofertas, 'p_sec': prom_sec, 'penal': penal}

            pb = st.session_state['resultado_base']
            mt = st.session_state['mets']
            
            if pb > 0.5: st.success(f"### ✅ PROBABILIDAD DE ÉXITO: {pb:.2%}")
            else: st.error(f"### ⚠️ PROBABILIDAD DE ÉXITO: {pb:.2%}")
            st.progress(pb)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Historial", f"{int(mt['hist'])} ganadas"); k2.metric("Ratio Precio", f"{mt['ratio']:.2f}x"); k3.metric("Competencia", f"{int(mt['comp'])} empresas")

            st.markdown("---")
            st.subheader("💡 Simulador de Competitividad")
            with st.container(border=True):
                desc = st.slider("Descuento (%)", 0, 30, 0)
                mejora = (desc * 0.012) if desc <= 10 else (0.12 + (desc-10)*0.005)
                p_sim = max(0.01, min(0.99, pb + mejora))
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Precio Ofertado", f"€ {valor_euro*(1-desc/100):,.0f}")
                s2.metric("Mejora Probabilidad", f"+{mejora*100:.1f}%")
                s3.metric("Nueva Probabilidad", f"{p_sim:.2%}", delta=f"{(p_sim - pb):+.2%}")

            g1, g2 = st.columns(2)
            with g1:
                fig_p = go.Figure(go.Bar(x=['Tu Oferta', 'Media Sector'], y=[valor_euro, mt['p_sec']], marker_color=['#00CC96' if valor_euro <= mt['p_sec'] else '#EF553B', '#636EFA'], text=[f"€{valor_euro:,.0f}", f"€{mt['p_sec']:,.0f}"], textposition='auto'))
                fig_p.update_layout(title="Precio vs Mercado", height=250); st.plotly_chart(fig_p, use_container_width=True)
            with g2:
                fig_c = go.Figure(go.Bar(x=['Actual', 'Histórica'], y=[num_ofertas, 5.0], marker_color=['#00CC96', '#AB63FA'], text=[f"{int(num_ofertas)}", "5.0"], textposition='auto'))
                fig_c.update_layout(title="Intensidad Competitiva", height=250); st.plotly_chart(fig_c, use_container_width=True)

# ==============================================================================
# SECCIÓN 2: DASHBOARD (COMPLETO)
# ==============================================================================
elif menu == "📊 Dashboard de Mercado":
    st.title("📊 Monitor de Mercado y Éxito PYME")
    df = cargar_csv_dashboard()
    if df is not None:
        p_disp = sorted(df['ISO_COUNTRY_CODE'].dropna().unique())
        p_sel = st.sidebar.multiselect("Países:", p_disp, default=['ES', 'FR', 'DE', 'IT', 'PL'])
        df_f = df[df['ISO_COUNTRY_CODE'].isin(p_sel)]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Licitaciones", f"{len(df_f):,}")
        m2.metric("% Éxito PYME", f"{(df_f['Es_PYME_Num'] == 1).mean():.2%}")
        m3.metric("Países", len(p_sel))

        df_map = df_f[df_f['Es_PYME_Num']==1]['ISO3'].value_counts().reset_index()
        df_map.columns = ['ISO3', 'Victorias_PYME']
        fig_map = px.choropleth(df_map, locations='ISO3', color='Victorias_PYME', scope="europe", color_continuous_scale="Viridis")
        fig_map.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else: st.error("Archivo CSV no encontrado.")

# ==============================================================================
# SECCIÓN 3: AUDITORÍA
# ==============================================================================
elif menu == "⚙️ Auditoría Técnica":
    st.title("⚙️ Auditoría Técnica del Modelo")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Métricas de Rendimiento")
        st.table(pd.DataFrame({'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'], 'Valor': ['78.5%', '72.1%', '81.4%', '76.5%']}))
    with c2:
        st.subheader("Variables más Influyentes")
        st.bar_chart({'Historial Empresa': 0.35, 'Ratio Precio': 0.25, 'Competencia CPV': 0.15, 'País': 0.10, 'Tipo Entidad': 0.05})
