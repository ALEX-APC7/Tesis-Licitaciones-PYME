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
# 0. PARCHE MAESTRO DE COMPATIBILIDAD PRO (OBLIGATORIO PARA RENDER)
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
# 1. CONFIGURACIÓN Y CARGA DE DATOS (MODELO + CSV DESDE DRIVE)
# ==============================================================================
st.set_page_config(page_title="Sistema de Licitaciones", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

ISO_2_TO_3 = {'ES': 'ESP', 'FR': 'FRA', 'DE': 'DEU', 'IT': 'ITA', 'PL': 'POL', 'PT': 'PRT', 'NL': 'NLD', 'BE': 'BEL', 'AT': 'AUT', 'SE': 'SWE', 'DK': 'DNK', 'FI': 'FIN', 'GR': 'GRC', 'IE': 'IRL', 'CZ': 'CZE'}
MAP_CONTRATO = {'W': 'Obras', 'U': 'Suministros', 'S': 'Servicios'}
MAP_PAIS = {'ES': 'España', 'FR': 'Francia', 'DE': 'Alemania', 'PL': 'Polonia', 'IT': 'Italia', 'PT': 'Portugal', 'NL': 'Países Bajos', 'BE': 'Bélgica'}
MAP_PROCEDIMIENTO = {'OPE': 'Abierto', 'RES': 'Restringido', 'NEG': 'Negociado', 'COMP': 'Competitivo'}
MAP_CRITERIO = {'L': 'Precio más bajo', 'M': 'Mejor Relación (MEAT)', 'O': 'Mixto'}
MAP_ENTIDAD = {'1': 'Gobierno Central', '3': 'Autoridades Locales', '6': 'Organismos Públicos', '8': 'Otras', 'Z': 'No especificado'}
MAP_ACTIVIDAD = {'Health': 'Salud', 'Education': 'Educación', 'Defence': 'Defensa', 'General public services': 'Servicios Públicos', 'Public order and safety': 'Seguridad', 'Environment': 'Medio Ambiente', 'Economic and financial affairs': 'Economía', 'Housing and community amenities': 'Vivienda', 'Social protection': 'Prot. Social', 'Recreation, culture and religion': 'Cultura', 'Other': 'Otra'}

@st.cache_resource
def cargar_recursos():
    # Modelo Joblib
    ID_MOD = "1jOCGQTRZfNNoF1kGHD_S6OAxgUkLmC6c"
    if not os.path.exists('datos_tesis.joblib'):
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(f"https://drive.google.com/uc?export=download&id={ID_MOD}", context=context) as response, open('datos_tesis.joblib', 'wb') as f:
            f.write(response.read())
    return joblib.load('datos_tesis.joblib')

@st.cache_data
def cargar_csv_dashboard():
    # CSV Bdatos_13_variables / export_CAN
    ID_CSV = "14PRk0KYhlxrtDsGFoXEw_giPCNbXTaug"
    if not os.path.exists('export_CAN_2023.csv'):
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(f"https://drive.google.com/uc?export=download&id={ID_CSV}", context=context) as response, open('export_CAN_2023.csv', 'wb') as f:
                f.write(response.read())
        except: return None
    
    df = pd.read_csv('export_CAN_2023.csv', low_memory=False)
    
    def buscar_col(keywords):
        for k in keywords:
            for c in df.columns:
                if k.upper() in str(c).upper(): return c
        return None

    c_pyme = buscar_col(['Es_PYME_Ganadora', 'B_CONTRACTOR_SME', 'WINNER_SME'])
    c_pais = buscar_col(['ISO_COUNTRY_CODE', 'COUNTRY', 'PAIS'])
    c_valor = buscar_col(['VALUE_EURO', 'VALOR', 'AMOUNT'])
    c_tipo = buscar_col(['TYPE_OF_CONTRACT', 'TIPO_CONTRATO'])

    if c_pyme: df['Es_PYME'] = df[c_pyme].astype(str).str[0].map({'Y': 'PYME', '1': 'PYME', 'N': 'NO PYME', '0': 'NO PYME'}).fillna('NO PYME')
    if c_pais: 
        df['ISO_COUNTRY_CODE'] = df[c_pais]
        df['ISO3'] = df['ISO_COUNTRY_CODE'].map(ISO_2_TO_3)
    if c_valor: df['VALUE_EURO'] = pd.to_numeric(df[c_valor], errors='coerce').fillna(0)
    if c_tipo: df['Tipo_Contrato_L'] = df[c_tipo].map(MAP_CONTRATO).fillna('Otro')
    return df

sistema = cargar_recursos()
modelo = sistema['modelo_entrenado']
ref_participacion = sistema.get('ref_participacion', {})
ref_promedio_precio = sistema.get('ref_promedio_sector', {})
ref_total_licitaciones = sistema.get('ref_total_licitaciones', {})
ref_promedio_competidores = sistema.get('ref_promedio_competidores', {})

# ==============================================================================
# 2. MENÚ DE NAVEGACIÓN
# ==============================================================================
with st.sidebar:
    st.title("🏛️ Menú Principal")
    menu = st.radio("Sección:", ["🚀 Simulador de Viabilidad", "📊 Dashboard de Mercado", "⚙️ Auditoría Técnica"])
    st.divider()
    st.caption("Tesis IT Ingeniería")

# ==============================================================================
# SECCIÓN 1: SIMULADOR (CÓDIGO INAMOVIBLE Y COMPLETO)
# ==============================================================================
if menu == "🚀 Simulador de Viabilidad":
    def membership_trapezoidal(x, a, b, c, d):
        if x <= a or x >= d: return 0.0
        if a < x < b: return (x - a) / (b - a)
        if b <= x <= c: return 1.0
        if c < x < d: return (d - x) / (d - c)
        return 0.0

    def membership_triangular(x, a, b, c):
        return max(min((x - a) / (b - a), (c - x) / (c - b)), 0) if b != a and c != b else 0.0

    st.title("🏛️ Sistema de Viabilidad de Licitaciones")
    st.markdown("**Análisis inteligente para PYMES en el mercado europeo**")
    st.markdown("---")

    if 'analisis_realizado' not in st.session_state: st.session_state['analisis_realizado'] = False
    def resetear(): 
        st.session_state['analisis_realizado'] = False
        if 'resultado_base' in st.session_state: del st.session_state['resultado_base']

    col_panel, col_result = st.columns([1, 1.5], gap="large")

    with col_panel:
        st.subheader("1. Datos del Proyecto")
        st.markdown("##### 💶 Variable Económica")
        valor_euro = st.number_input("Valor de tu Oferta (€)", min_value=0.0, value=150000.0, step=5000.0, on_change=resetear)
        num_ofertas = st.number_input("Competencia Estimada (Nº Empresas)", min_value=1, value=3, on_change=resetear)
        st.markdown("##### 📋 Detalles Técnicos")
        cpv_code = st.text_input("Código CPV", value="45000000", on_change=resetear)
        pais = st.selectbox("País", options=list(MAP_PAIS.keys()), format_func=lambda x: MAP_PAIS[x], on_change=resetear)
        tipo_contrato = st.selectbox("Tipo Contrato", options=['W', 'U', 'S'], format_func=lambda x: MAP_CONTRATO[x], on_change=resetear)
        tipo_proc = st.selectbox("Procedimiento", options=['OPE', 'RES', 'NEG', 'COMP'], format_func=lambda x: MAP_PROCEDIMIENTO.get(x, x), on_change=resetear)
        criterio = st.selectbox("Criterio", options=['L', 'M', 'O'], format_func=lambda x: MAP_CRITERIO[x], on_change=resetear)
        tipo_entidad = st.selectbox("Entidad", options=['1', '3', '6', '8', 'Z'], format_func=lambda x: MAP_ENTIDAD[x], on_change=resetear)
        actividad = st.selectbox("Actividad", options=list(MAP_ACTIVIDAD.keys()), format_func=lambda x: MAP_ACTIVIDAD[x], on_change=resetear)
        st.markdown("##### 🏢 Tu Empresa")
        empresa = st.text_input("Nombre del Licitante", placeholder="Ej: Mi Empresa S.A.", on_change=resetear)
        st.button("🚀 Calcular Viabilidad", type="primary", use_container_width=True, on_click=lambda: st.session_state.update({'analisis_realizado': True}))

    if st.session_state['analisis_realizado']:
        with col_result:
            st.subheader("2. Resultados del Análisis")
            if 'resultado_base' not in st.session_state:
                with st.spinner('Procesando...'):
                    cpv_in = cpv_code.strip()
                    prom_sec = ref_promedio_precio.get(cpv_in, ref_promedio_precio.get(int(cpv_in) if cpv_in.isdigit() else None, valor_euro))
                    comp_media = ref_promedio_competidores.get(cpv_in, 5.0)
                    comp_tot = ref_total_licitaciones.get(cpv_in, 10)
                    hist = float(ref_participacion.get(empresa, 0))
                    ratio = float(valor_euro / (prom_sec if prom_sec != 0 else 1))

                    input_df = pd.DataFrame({'Valor_Estimado_EUR': [float(valor_euro)], 'Num_Ofertas_Recibidas': [float(num_ofertas)], 'Participacion_Historica_Empresa': [hist], 'Competencia_Sector_CPV': [float(comp_tot)], 'Ratio_Valor_Sector': [ratio], 'Codigo_CPV_Sector': [str(cpv_code)], 'ISO_COUNTRY_CODE': [str(pais)], 'TYPE_OF_CONTRACT': [str(tipo_contrato)], 'Tipo_Procedimiento': [str(tipo_proc)], 'MAIN_ACTIVITY': [str(actividad)], 'CRIT_CODE': [str(criterio)], 'CAE_TYPE': [str(tipo_entidad)]})
                    
                    prob_ml = modelo.predict_proba(input_df)[0][1]
                    mu_hist_nula = membership_trapezoidal(hist, -1, 0, 0, 5)
                    mu_precio_riesgo = min(1.0, (valor_euro - prom_sec) / prom_sec) if valor_euro > (prom_sec * 1.1) else 0.0

                    msgs = []
                    penal = 0.0
                    if mu_hist_nula > 0.5: penal += 0.125; msgs.append("📉 **Historial:** Sin adjudicaciones previas (-12.5%).")
                    if num_ofertas == 2: penal += 0.10; msgs.append("👥 **Competencia:** Penalización leve por 2 rivales (-10%).")
                    elif num_ofertas == 3: penal += 0.20; msgs.append("👥 **Competencia:** Dificultad alta (-20%).")
                    elif num_ofertas >= 4: penal += 0.25; msgs.append("⚠️ **Saturación:** Penalización máxima (-25%).")
                    if mu_precio_riesgo > 0.2: penal += 0.15; msgs.append("💰 **Precio:** Oferta por encima del promedio del sector (-15%).")

                    st.session_state['resultado_base'] = max(0.01, min(0.99, prob_ml - penal))
                    st.session_state['mensajes_base'] = msgs
                    st.session_state['metricas_base'] = {'hist': hist, 'ratio': ratio, 'comp': num_ofertas, 'prom_sec': prom_sec, 'penal': penal, 'comp_media': comp_media}

            pb = st.session_state['resultado_base']
            mets = st.session_state['metricas_base']
            if pb > 0.5: st.success(f"### ✅ PROBABILIDAD DE ÉXITO: {pb:.2%}")
            else: st.error(f"### ⚠️ PROBABILIDAD DE ÉXITO: {pb:.2%}")
            st.progress(pb)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Historial", f"{int(mets['hist'])} ganadas"); k2.metric("Ratio Precio", f"{mets['ratio']:.2f}x"); k3.metric("Competencia", f"{int(mets['comp'])} empresas")

            st.markdown("---")
            st.subheader("💡 Simulador de Competitividad")
            with st.container(border=True):
                val_desc = st.slider("Descuento a aplicar (%)", 0, 30, 0, key="sim_master")
                benef = (val_desc * 0.012) if val_desc <= 10 else (0.12 + (val_desc-10)*0.005)
                prob_sim = max(0.01, min(0.99, pb + benef))
                nuevo_p = valor_euro * (1 - (val_desc/100))
                s1, s2, s3 = st.columns(3)
                s1.metric("Precio Ofertado", f"€ {nuevo_p:,.0f}"); s2.metric("Mejora Probabilidad", f"+{benef*100:.1f}%"); s3.metric("Nueva Probabilidad", f"{prob_sim:.2%}", delta=f"{(prob_sim - pb):+.2%}")

            st.markdown("#### 📊 Benchmarking de Mercado")
            g1, g2 = st.columns(2)
            with g1:
                fig_p = go.Figure(go.Bar(x=['Tu Oferta', 'Promedio Sector'], y=[valor_euro, mets['prom_sec']], marker_color=['#00CC96' if valor_euro <= mets['prom_sec'] else '#EF553B', '#636EFA'], text=[f"€{valor_euro:,.0f}", f"€{mets['prom_sec']:,.0f}"], textposition='auto'))
                fig_p.update_layout(title="Competitividad Económica", height=300); st.plotly_chart(fig_p, use_container_width=True)
            with g2:
                fig_c = go.Figure(go.Bar(x=['Competencia Actual', 'Promedio Histórico'], y=[num_ofertas, mets['comp_media']], marker_color=['#00CC96', '#AB63FA'], text=[f"{int(num_ofertas)}", f"{mets['comp_media']:.1f}"], textposition='auto'))
                fig_c.update_layout(title="Intensidad Competitiva", height=300); st.plotly_chart(fig_c, use_container_width=True)

            with st.expander("📝 Factores de Riesgo Detectados", expanded=True):
                if not st.session_state['mensajes_base']: st.success("✅ Perfil competitivo.")
                else:
                    for msg in st.session_state['mensajes_base']: st.markdown(f"- {msg}")
                st.caption(f"Ajuste total aplicado por lógica difusa: -{mets['penal']*100:.1f}%")

# ==============================================================================
# SECCIÓN 2: DASHBOARD (MAPA + PIE + HISTOGRAMA)
# ==============================================================================
elif menu == "📊 Dashboard de Mercado":
    st.title("📊 Monitor de Mercado y Éxito PYME")
    df_raw = cargar_csv_dashboard()
    if df_raw is not None:
        p_sel = st.sidebar.multiselect("Países:", sorted(df_raw['ISO_COUNTRY_CODE'].dropna().unique()), default=['ES', 'FR', 'DE', 'IT', 'PL'])
        df_f = df_raw[df_raw['ISO_COUNTRY_CODE'].isin(p_sel)] if p_sel else df_raw
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Licitaciones (Filtro)", f"{len(df_f):,}"); m2.metric("% Éxito PYME", f"{(df_f['Es_PYME'] == 'PYME').mean():.2%}"); m3.metric("Países Visibles", len(p_sel))

        st.subheader("🌍 Mapa de Calor: Adjudicaciones a PYMES")
        df_map = df_f[df_f['Es_PYME']=='PYME']['ISO3'].value_counts().reset_index()
        df_map.columns = ['ISO3', 'Victorias_PYME']
        fig_map = px.choropleth(df_map, locations='ISO3', locationmode="ISO-3", color='Victorias_PYME', scope="europe", color_continuous_scale="Viridis")
        fig_map.update_layout(height=650, margin={"r":0,"t":30,"l":0,"b":0}); st.plotly_chart(fig_map, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("PYME vs NO PYME")
            fig_pie = px.pie(df_f, names='Es_PYME', color='Es_PYME', color_discrete_map={'PYME': '#00CC96', 'NO PYME': '#EF553B'}, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("💰 Distribución de Precios")
            fig_hist = px.histogram(df_f[df_f['VALUE_EURO'] < 2000000], x="VALUE_EURO", color="Es_PYME", barmode="overlay", color_discrete_map={'PYME': '#00CC96', 'NO PYME': '#EF553B'})
            st.plotly_chart(fig_hist, use_container_width=True)
    else: st.error("No se pudo cargar el archivo de datos.")

# ==============================================================================
# SECCIÓN 3: AUDITORÍA (TU VERSIÓN ORIGINAL)
# ==============================================================================
elif menu == "⚙️ Auditoría Técnica":
    st.title("⚙️ Auditoría Técnica del Modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Métricas de Rendimiento")
        dt = {'Métrica': ['Accuracy (Global)', 'Precision (PYME)', 'Recall (PYME)', 'F1-Score'], 'Valor': ['78.5%', '72.1%', '81.4%', '76.5%']}
        st.table(pd.DataFrame(dt))
        st.info("El modelo prioriza el Recall para no perder oportunidades de identificar una PYME real.")
    with col2:
        st.subheader("Variables más Influyentes")
        st.bar_chart({'Historial Empresa': 0.35, 'Ratio Precio': 0.25, 'Competencia CPV': 0.15, 'País': 0.10, 'Tipo Entidad': 0.05})
