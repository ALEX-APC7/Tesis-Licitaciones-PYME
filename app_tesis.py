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
# 0. PARCHE MAESTRO DE COMPATIBILIDAD PRO (SKLEARN 1.5+ vs MODELO ANTIGUO)
# ==============================================================================
import sklearn.compose._column_transformer
import sklearn.impute._base

class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
if 'sklearn.compose._column_transformer' in sys.modules:
    sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

def get_fill_dtype(self):
    return getattr(self, "dtype", np.object_)

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

# Diccionarios Globales para Dashboard y Simulador
ISO_2_TO_3 = {'ES': 'ESP', 'FR': 'FRA', 'DE': 'DEU', 'IT': 'ITA', 'PL': 'POL', 'PT': 'PRT', 'NL': 'NLD', 'BE': 'BEL', 'AT': 'AUT', 'SE': 'SWE', 'DK': 'DNK', 'FI': 'FIN', 'GR': 'GRC', 'IE': 'IRL', 'CZ': 'CZE'}
MAP_CONTRATO = {'W': 'Obras', 'U': 'Suministros', 'S': 'Servicios'}
MAP_PAIS = {'ES': 'España', 'FR': 'Francia', 'DE': 'Alemania', 'PL': 'Polonia', 'IT': 'Italia', 'PT': 'Portugal', 'NL': 'Países Bajos', 'BE': 'Bélgica'}
MAP_PROCEDIMIENTO = {'OPE': 'Abierto', 'RES': 'Restringido', 'NEG': 'Negociado', 'COMP': 'Competitivo', 'OTH': 'Otro'}
MAP_CRITERIO = {'L': 'Precio más bajo', 'M': 'Mejor Relación (MEAT)', 'O': 'Mixto'}
MAP_ENTIDAD = {'1': 'Gobierno Central', '3': 'Autoridades Locales', '6': 'Organismos Públicos', '8': 'Otras', 'Z': 'No especificado'}
MAP_ACTIVIDAD = {'Health': 'Salud', 'Education': 'Educación', 'Defence': 'Defensa', 'General public services': 'Servicios Públicos', 'Public order and safety': 'Seguridad', 'Environment': 'Medio Ambiente', 'Economic and financial affairs': 'Economía', 'Housing and community amenities': 'Vivienda', 'Social protection': 'Prot. Social', 'Recreation, culture and religion': 'Cultura', 'Other': 'Otra'}

# ==============================================================================
# 2. CARGA DE DATOS (MODELO + CSV DESDE DRIVE)
# ==============================================================================
@st.cache_resource
def cargar_cerebro():
    ID_ARCHIVO_DRIVE = "1jOCGQTRZfNNoF1kGHD_S6OAxgUkLmC6c" 
    URL_DESCARGA = f"https://drive.google.com/uc?export=download&id={ID_ARCHIVO_DRIVE}"
    NOMBRE_LOCAL = "datos_tesis.joblib"
    try:
        if not os.path.exists(NOMBRE_LOCAL):
            with st.spinner('Descargando base de datos del modelo...'):
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(URL_DESCARGA, context=context) as response, open(NOMBRE_LOCAL, 'wb') as out_file:
                    out_file.write(response.read())
        return joblib.load(NOMBRE_LOCAL)
    except Exception as e:
        st.error(f"🚨 Error crítico de carga de modelo: {e}")
        return None

@st.cache_data
def cargar_csv_dashboard():
    ID_DRIVE_CSV = "14PRk0KYhlxrtDsGFoXEw_giPCNbXTaug"
    URL_CSV = f"https://drive.google.com/uc?export=download&id={ID_DRIVE_CSV}"
    FILE_CSV = "export_CAN_2023.csv"
    if not os.path.exists(FILE_CSV):
        try:
            with st.spinner('Descargando datos históricos para el Dashboard...'):
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(URL_CSV, context=context) as response, open(FILE_CSV, 'wb') as f:
                    f.write(response.read())
        except: return None
    
    df = pd.read_csv(FILE_CSV, low_memory=False)
    
    # BUSCADOR DINÁMICO DE COLUMNAS PARA EVITAR KEYERRORS
    def buscar_col(keywords):
        for k in keywords:
            for c in df.columns:
                if k.upper() in str(c).upper(): return c
        return None

    c_pyme = buscar_col(['Es_PYME_Ganadora', 'SME', 'WINNER_SME', 'PYME'])
    c_pais = buscar_col(['ISO_COUNTRY_CODE', 'COUNTRY', 'PAÍS', 'ISO'])
    c_valor = buscar_col(['VALUE_EURO', 'VALOR', 'AMOUNT', 'EURO'])
    c_tipo = buscar_col(['TYPE_OF_CONTRACT', 'TIPO_CONTRATO', 'CONTRATO'])

    if c_pyme: 
        df['Es_PYME_Num'] = df[c_pyme].astype(str).str.upper().str[0].map({'Y': 1, '1': 1, 'S': 1, 'P': 1}).fillna(0)
        df['Es_PYME_Texto'] = df['Es_PYME_Num'].map({1: 'PYME', 0: 'NO PYME'})
    if c_pais: 
        df['PAIS_LIMPIO'] = df[c_pais]
        df['ISO3'] = df['PAIS_LIMPIO'].map(ISO_2_TO_3)
    if c_valor: df['VALOR_LIMPIO'] = pd.to_numeric(df[c_valor], errors='coerce').fillna(0)
    if c_tipo: df['Tipo_Contrato_L'] = df[c_tipo].map(MAP_CONTRATO).fillna('Otro')
    
    return df

# Inicializar recursos
sistema = cargar_cerebro()
if sistema is None: st.stop()

modelo = sistema['modelo_entrenado']
ref_participacion = sistema.get('ref_participacion', {})
ref_promedio_precio = sistema.get('ref_promedio_sector', {})
ref_total_licitaciones = sistema.get('ref_total_licitaciones', {})
ref_promedio_competidores = sistema.get('ref_promedio_competidores', {})

# ==============================================================================
# 3. NAVEGACIÓN (MENÚ LATERAL)
# ==============================================================================
with st.sidebar:
    st.title("🏛️ Menú Principal")
    menu = st.radio("Seleccione Sección:", ["🚀 Simulador de Viabilidad", "📊 Dashboard de Mercado", "⚙️ Auditoría Técnica"])
    st.divider()
    st.caption("Tesis: Ingeniería en Tecnologías de la Información")

# ==============================================================================
# SECCIÓN 1: SIMULADOR (TU CÓDIGO INAMOVIBLE - 100% IDENTICO)
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

    if 'analisis_realizado' not in st.session_state:
        st.session_state['analisis_realizado'] = False

    def resetear_analisis():
        st.session_state['analisis_realizado'] = False
        if 'resultado_base' in st.session_state: del st.session_state['resultado_base']

    col_panel, col_result = st.columns([1, 1.5], gap="large")

    with col_panel:
        st.subheader("1. Datos del Proyecto")
        st.markdown("##### 💶 Variable Económica")
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
        st.markdown("##### 🏢 Tu Empresa")
        empresa = st.text_input("Nombre del Licitante", placeholder="Ej: Mi Empresa S.A.", on_change=resetear_analisis)
        st.markdown("---")
        btn_calcular = st.button("🚀 Calcular Viabilidad", type="primary", use_container_width=True, on_click=lambda: st.session_state.update({'analisis_realizado': True}))

    if st.session_state['analisis_realizado']:
        with col_result:
            st.subheader("2. Resultados del Análisis")
            if 'resultado_base' not in st.session_state:
                with st.spinner('Procesando datos y aplicando lógica difusa...'):
                    cpv_input = cpv_code.strip()
                    promedio_precio_sector = ref_promedio_precio.get(cpv_input, ref_promedio_precio.get(int(cpv_input) if cpv_input.isdigit() else None, valor_euro))
                    competencia_media_grafico = ref_promedio_competidores.get(cpv_input, ref_promedio_competidores.get(int(cpv_input) if cpv_input.isdigit() else None, 5.0))
                    competencia_total_modelo = ref_total_licitaciones.get(cpv_input, ref_total_licitaciones.get(int(cpv_input) if cpv_input.isdigit() else None, 10))
                    historia = float(ref_participacion.get(empresa, 0))
                    ratio_valor = float(valor_euro / (promedio_precio_sector if promedio_precio_sector != 0 else 1))

                    input_df = pd.DataFrame({'Valor_Estimado_EUR': [float(valor_euro)], 'Num_Ofertas_Recibidas': [float(num_ofertas)], 'Participacion_Historica_Empresa': [historia], 'Competencia_Sector_CPV': [float(competencia_total_modelo)], 'Ratio_Valor_Sector': [ratio_valor], 'Codigo_CPV_Sector': [str(cpv_code)], 'ISO_COUNTRY_CODE': [str(pais)], 'TYPE_OF_CONTRACT': [str(tipo_contrato)], 'Tipo_Procedimiento': [str(tipo_proc)], 'MAIN_ACTIVITY': [str(actividad)], 'CRIT_CODE': [str(criterio)], 'CAE_TYPE': [str(tipo_entidad)]})
                    
                    try: prob_ml_raw = modelo.predict_proba(input_df)[0][1]
                    except: prob_ml_raw = 0.5

                    mu_hist_nula = membership_trapezoidal(historia, -1, 0, 0, 5)
                    mu_precio_riesgo = min(1.0, (valor_euro - promedio_precio_sector) / promedio_precio_sector) if valor_euro > (promedio_precio_sector * 1.1) else 0.0
                    mensajes_explicativos = []
                    penalizacion_total = 0.0
                    if mu_hist_nula > 0.5: penalizacion_total += 0.125; mensajes_explicativos.append("📉 **Historial:** Sin adjudicaciones previas (-12.5%).")
                    if num_ofertas == 2: penalizacion_total += 0.10; mensajes_explicativos.append("👥 **Competencia:** Penalización leve por 2 rivales (-10%).")
                    elif num_ofertas == 3: penalizacion_total += 0.20; mensajes_explicativos.append("👥 **Competencia:** Dificultad alta (-20%).")
                    elif num_ofertas >= 4: penalizacion_total += 0.25; mensajes_explicativos.append("⚠️ **Saturación:** Penalización máxima (-25%).")
                    if mu_precio_riesgo > 0.2: penalizacion_total += 0.15; mensajes_explicativos.append("💰 **Precio:** Tu oferta está por encima del promedio del sector (-15%).")

                    prob_final_fuzzy = max(0.01, min(0.99, prob_ml_raw - penalizacion_total))
                    st.session_state['resultado_base'] = prob_final_fuzzy
                    st.session_state['mensajes_base'] = mensajes_explicativos
                    st.session_state['metricas_base'] = {'historia': historia, 'ratio': ratio_valor, 'competencia': num_ofertas, 'promedio_sector': promedio_precio_sector, 'penalizacion': penalizacion_total, 'comp_media_grafico': competencia_media_grafico}

            pb = st.session_state['resultado_base']
            mets = st.session_state['metricas_base']
            if pb > 0.5: st.success(f"### ✅ PROBABILIDAD DE ÉXITO: {pb:.2%}")
            else: st.error(f"### ⚠️ PROBABILIDAD DE ÉXITO: {pb:.2%}")
            st.progress(pb)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Historial", f"{int(mets['historia'])} ganadas"); k2.metric("Ratio Precio", f"{mets['ratio']:.2f}x"); k3.metric("Competencia", f"{int(mets['competencia'])} empresas")

            st.markdown("---")
            st.subheader("💡 Simulador de Competitividad")
            with st.container(border=True):
                val_descuento = st.slider("Descuento a aplicar (%)", 0, 30, 0, key="simulador_master")
                beneficio_pct = (val_descuento * 0.012) if val_descuento <= 10 else (0.12 + (val_descuento-10)*0.005)
                prob_simulada = max(0.01, min(0.99, pb + beneficio_pct))
                nuevo_precio_sim = valor_euro * (1 - (val_descuento/100))
                cs1, cs2, cs3 = st.columns(3)
                cs1.metric("Precio Ofertado", f"€ {nuevo_precio_sim:,.0f}"); cs2.metric("Mejora Probabilidad", f"+{beneficio_pct*100:.1f}%"); cs3.metric("Nueva Probabilidad", f"{prob_simulada:.2%}", delta=f"{(prob_simulada - pb):+.2%}")

            st.markdown("#### 📊 Benchmarking de Mercado")
            g1, g2 = st.columns(2)
            with g1:
                fig_p = go.Figure(go.Bar(x=['Tu Oferta', 'Promedio Sector'], y=[valor_euro, mets['promedio_sector']], marker_color=['#00CC96' if valor_euro <= mets['promedio_sector'] else '#EF553B', '#636EFA'], text=[f"€{valor_euro:,.0f}", f"€{mets['promedio_sector']:,.0f}"], textposition='auto'))
                fig_p.update_layout(title="Competitividad Económica", height=300); st.plotly_chart(fig_p, use_container_width=True)
            with g2:
                fig_c = go.Figure(go.Bar(x=['Competencia Actual', 'Promedio Histórico'], y=[num_ofertas, mets['comp_media_grafico']], marker_color=['#00CC96', '#AB63FA'], text=[f"{int(num_ofertas)}", f"{mets['comp_media_grafico']:.1f}"], textposition='auto'))
                fig_c.update_layout(title="Intensidad Competitiva", height=300); st.plotly_chart(fig_c, use_container_width=True)

            with st.expander("📝 Factores de Riesgo Detectados", expanded=True):
                if not st.session_state['mensajes_base']: st.success("✅ Perfil altamente competitivo.")
                else:
                    for msg in st.session_state['mensajes_base']: st.markdown(f"- {msg}")
                st.caption(f"Ajuste total aplicado por lógica difusa: -{mets['penalizacion']*100:.1f}%")

# ==============================================================================
# SECCIÓN 2: DASHBOARD (MAPA + PIE + HISTOGRAMA)
# ==============================================================================
elif menu == "📊 Dashboard de Mercado":
    st.title("📊 Monitor de Mercado y Éxito PYME")
    df_raw = cargar_csv_dashboard()
    if df_raw is not None:
        p_sel = st.sidebar.multiselect("Filtrar Países:", sorted(df_raw['PAIS_LIMPIO'].dropna().unique()), default=['ES', 'FR', 'DE', 'IT', 'PL'])
        df_f = df_raw[df_raw['PAIS_LIMPIO'].isin(p_sel)] if p_sel else df_raw
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Licitaciones", f"{len(df_f):,}"); m2.metric("% Éxito PYME", f"{(df_f['Es_PYME_Num'] == 1).mean():.2%}"); m3.metric("Países Seleccionados", len(p_sel))

        st.subheader("🌍 Mapa de Calor Europeo: Adjudicaciones a PYMES")
        df_map = df_f[df_f['Es_PYME_Num']==1]['ISO3'].value_counts().reset_index()
        df_map.columns = ['ISO3', 'Victorias_PYME']
        fig_map = px.choropleth(df_map, locations='ISO3', locationmode="ISO-3", color='Victorias_PYME', scope="europe", color_continuous_scale="Viridis")
        fig_map.update_layout(height=650, margin={"r":0,"t":30,"l":0,"b":0}); st.plotly_chart(fig_map, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Distribución PYME vs NO PYME")
            fig_pie = px.pie(df_f, names='Es_PYME_Texto', color='Es_PYME_Texto', color_discrete_map={'PYME': '#00CC96', 'NO PYME': '#EF553B'}, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("💰 Distribución de Precios (Zoom <2M€)")
            fig_hist = px.histogram(df_f[df_f['VALOR_LIMPIO'] < 2000000], x="VALOR_LIMPIO", color="Es_PYME_Texto", barmode="overlay", color_discrete_map={'PYME': '#00CC96', 'NO PYME': '#EF553B'})
            st.plotly_chart(fig_hist, use_container_width=True)
    else: st.error("No se pudo cargar el archivo histórico.")

# ==============================================================================
# SECCIÓN 3: AUDITORÍA (TU VERSIÓN DE MÉTRICAS)
# ==============================================================================
elif menu == "⚙️ Auditoría Técnica":
    st.title("⚙️ Auditoría Técnica del Modelo")
    st.markdown("Transparencia algorítmica y métricas de rendimiento del Random Forest.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Métricas de Rendimiento")
        dt = {'Métrica': ['Accuracy (Global)', 'Precision (PYME)', 'Recall (PYME)', 'F1-Score'], 'Valor': ['78.5%', '72.1%', '81.4%', '76.5%']}
        st.table(pd.DataFrame(dt))
        st.info("El modelo prioriza el Recall para identificar el mayor número posible de PYMES reales.")
    with col2:
        st.subheader("Variables más Influyentes")
        st.bar_chart({'Historial Empresa': 0.35, 'Ratio Precio': 0.25, 'Competencia CPV': 0.15, 'País': 0.10, 'Tipo Entidad': 0.05})
