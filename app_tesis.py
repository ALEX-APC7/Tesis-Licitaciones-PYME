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

class _RemainderColsList(list): pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
if 'sklearn.compose._column_transformer' in sys.modules:
    sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

def get_fill_dtype(self): return getattr(self, "dtype", np.object_)
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

# Diccionarios de Mapeo
ISO_2_TO_3 = {'ES': 'ESP', 'FR': 'FRA', 'DE': 'DEU', 'IT': 'ITA', 'PL': 'POL', 'PT': 'PRT', 'NL': 'NLD', 'BE': 'BEL'}
MAP_CONTRATO = {'W': 'Obras', 'U': 'Suministros', 'S': 'Servicios'}
MAP_PROCEDIMIENTO = {'OPE': 'Abierto', 'RES': 'Restringido', 'NEG': 'Negociado', 'COMP': 'Competitivo', 'OTH': 'Otro'}
MAP_CRITERIO = {'L': 'Precio más bajo', 'M': 'Mejor Relación (MEAT)', 'O': 'Mixto'}
MAP_ENTIDAD = {'1': 'Gobierno Central', '3': 'Autoridades Locales', '6': 'Organismos Públicos', '8': 'Otras', 'Z': 'No especificado'}
MAP_ACTIVIDAD = {'Health': 'Salud', 'Education': 'Educación', 'Defence': 'Defensa', 'General public services': 'Servicios Públicos', 'Public order and safety': 'Seguridad', 'Environment': 'Medio Ambiente', 'Economic and financial affairs': 'Economía', 'Housing and community amenities': 'Vivienda', 'Social protection': 'Prot. Social', 'Recreation, culture and religion': 'Cultura', 'Other': 'Otra'}
MAP_PAIS = {'ES': 'España', 'FR': 'Francia', 'DE': 'Alemania', 'PL': 'Polonia', 'IT': 'Italia', 'PT': 'Portugal', 'NL': 'Países Bajos', 'BE': 'Bélgica'}

# ==============================================================================
# 2. CARGA DE RECURSOS (MODELO + CSV)
# ==============================================================================
@st.cache_resource
def cargar_cerebro():
    ID_ARCHIVO_DRIVE = "1jOCGQTRZfNNoF1kGHD_S6OAxgUkLmC6c" 
    URL_DESCARGA = f"https://drive.google.com/uc?export=download&id={ID_ARCHIVO_DRIVE}"
    NOMBRE_LOCAL = "datos_tesis.joblib"
    try:
        if not os.path.exists(NOMBRE_LOCAL):
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(URL_DESCARGA, context=context) as response, open(NOMBRE_LOCAL, 'wb') as out_file:
                out_file.write(response.read())
        return joblib.load(NOMBRE_LOCAL)
    except Exception as e:
        st.error(f"🚨 Error crítico de carga del modelo: {e}")
        return None

@st.cache_data
def cargar_datos_csv():
    ruta = 'export_CAN_2023.csv'
    if os.path.exists(ruta):
        df = pd.read_csv(ruta, low_memory=False)
        col_target = 'B_CONTRACTOR_SME' if 'B_CONTRACTOR_SME' in df.columns else 'WINNER_SME'
        df['Es_PYME'] = df[col_target].fillna('N').map({'Y': 'PYME', 'N': 'NO PYME'})
        df['ISO3'] = df['ISO_COUNTRY_CODE'].map(ISO_2_TO_3)
        df['Tipo_Contrato'] = df['TYPE_OF_CONTRACT'].map(MAP_CONTRATO).fillna('Otro')
        return df
    return None

sistema = cargar_cerebro()
if sistema is None: st.stop()

modelo = sistema['modelo_entrenado']
ref_participacion = sistema['ref_participacion']
ref_promedio_precio = sistema.get('ref_promedio_sector', sistema.get('ref_promedio_precio', {}))
ref_total_licitaciones = sistema.get('ref_total_licitaciones', sistema.get('ref_competencia_cpv', {}))
ref_promedio_competidores = sistema.get('ref_promedio_competidores', {})

# ==============================================================================
# 3. NAVEGACIÓN LATERAL
# ==============================================================================
with st.sidebar:
    st.title("🏛️ Menú Principal")
    seccion = st.radio("Seleccione Vista:", ["🚀 Simulador de Viabilidad", "📊 Dashboard de Mercado", "⚙️ Auditoría Técnica"])
    st.divider()
    st.info("Tesis de Ingeniería IT\nAnálisis de PYMES en la UE")

# ==============================================================================
# SECCIÓN 1: SIMULADOR (TU CÓDIGO ORIGINAL SIN CAMBIOS)
# ==============================================================================
if seccion == "🚀 Simulador de Viabilidad":
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
                    prob_ml_raw = modelo.predict_proba(input_df)[0][1]
                    mu_hist_nula = membership_trapezoidal(historia, -1, 0, 0, 5)
                    mu_precio_riesgo = min(1.0, (valor_euro - promedio_precio_sector) / promedio_precio_sector) if valor_euro > (promedio_precio_sector * 1.1) else 0.0
                    mensajes_explicativos = []
                    penalizacion_total = 0.0
                    if mu_hist_nula > 0.5: penalizacion_total += 0.125; mensajes_explicativos.append("📉 **Historial:** Sin adjudicaciones previas (-12.5%).")
                    if num_ofertas == 2: penalizacion_total += 0.10; mensajes_explicativos.append("👥 **Competencia:** Penalización leve por 2 rivales (-10%).")
                    elif num_ofertas == 3: penalizacion_total += 0.20; mensajes_explicativos.append("👥 **Competencia:** Dificultad alta (-20%).")
                    elif num_ofertas >= 4: penalizacion_total += 0.25; mensajes_explicativos.append("⚠️ **Saturación:** Penalización máxima (-25%).")
                    if mu_precio_riesgo > 0.2: penalizacion_total += 0.15; mensajes_explicativos.append("💰 **Precio:** Oferta por encima del promedio (-15%).")
                    prob_final_fuzzy = max(0.01, min(0.99, prob_ml_raw - penalizacion_total))
                    st.session_state['resultado_base'] = prob_final_fuzzy
                    st.session_state['mensajes_base'] = mensajes_explicativos
                    st.session_state['metricas_base'] = {'historia': historia, 'ratio': ratio_valor, 'competencia': num_ofertas, 'promedio_sector': promedio_precio_sector, 'penalizacion': penalizacion_total, 'comp_media_grafico': competencia_media_grafico}

            prob_base = st.session_state['resultado_base']
            msgs = st.session_state['mensajes_base']
            mets = st.session_state['metricas_base']
            if prob_base > 0.5: st.success(f"### ✅ PROBABILIDAD DE ÉXITO: {prob_base:.2%}")
            else: st.error(f"### ⚠️ PROBABILIDAD DE ÉXITO: {prob_base:.2%}")
            st.progress(prob_base)
            k1, k2, k3 = st.columns(3)
            k1.metric("Historial", f"{int(mets['historia'])} ganadas"); k2.metric("Ratio Precio", f"{mets['ratio']:.2f}x"); k3.metric("Competencia", f"{int(mets['competencia'])} empresas")
            st.markdown("---")
            st.subheader("💡 Simulador de Competitividad")
            with st.container(border=True):
                val_descuento = st.slider("Descuento a aplicar (%)", 0, 30, 0, key="sim_libre")
                beneficio_pct = (val_descuento * 0.012) if val_descuento <= 10 else (0.12 + (val_descuento-10)*0.005)
                prob_simulada = max(0.01, min(0.99, prob_base + beneficio_pct))
                st.metric("Nueva Probabilidad", f"{prob_simulada:.2%}", delta=f"{(prob_simulada - prob_base):+.2%}")
            st.markdown("#### 📊 Benchmarking de Mercado")
            g1, g2 = st.columns(2)
            with g1:
                fig_p = go.Figure(go.Bar(x=['Tu Oferta', 'Promedio Sector'], y=[valor_euro, mets['promedio_sector']], marker_color=['#00CC96' if valor_euro <= mets['promedio_sector'] else '#EF553B', '#636EFA'], text=[f"€{valor_euro:,.0f}", f"€{mets['promedio_sector']:,.0f}"], textposition='auto'))
                fig_p.update_layout(title="Precio vs Mercado", height=300); st.plotly_chart(fig_p, use_container_width=True)
            with g2:
                fig_c = go.Figure(go.Bar(x=['Actual', 'Histórica'], y=[num_ofertas, mets['comp_media_grafico']], marker_color=['#00CC96', '#AB63FA'], text=[f"{int(num_ofertas)}", f"{mets['comp_media_grafico']:.1f}"], textposition='auto'))
                fig_c.update_layout(title="Intensidad Competitiva", height=300); st.plotly_chart(fig_c, use_container_width=True)
            with st.expander("📝 Factores de Riesgo"):
                for m in msgs: st.markdown(f"- {m}")
                st.caption(f"Ajuste total: -{mets['penalizacion']*100:.1f}%")

# ==============================================================================
# SECCIÓN 2: DASHBOARD (TU CÓDIGO ORIGINAL)
# ==============================================================================
elif seccion == "📊 Dashboard de Mercado":
    st.title("📊 Monitor de Mercado y Éxito PYME")
    df_raw = cargar_datos_csv()
    if df_raw is not None:
        st.sidebar.header("🔍 Filtros Dinámicos")
        paises_sel = st.sidebar.multiselect("Seleccionar Países:", options=sorted(df_raw['ISO_COUNTRY_CODE'].dropna().unique()), default=['ES', 'FR', 'DE', 'IT', 'PL'])
        df_f = df_raw[df_raw['ISO_COUNTRY_CODE'].isin(paises_sel)] if paises_sel else df_raw
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Licitaciones", f"{len(df_f):,}")
        k2.metric("% Éxito PYME", f"{(df_f['Es_PYME'] == 'PYME').mean():.2%}")
        k3.metric("Países", len(paises_sel))
        
        st.subheader("🌍 Mapa de Calor: Adjudicaciones a PYMES")
        df_map = df_f[df_f['Es_PYME']=='PYME']['ISO3'].value_counts().reset_index()
        df_map.columns = ['ISO3', 'Victorias_PYME']
        fig_map = px.choropleth(df_map, locations='ISO3', locationmode="ISO-3", color='Victorias_PYME', scope="europe", color_continuous_scale="Viridis")
        fig_map.update_layout(height=650, margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df_f, names='Es_PYME', title="PYME vs NO PYME", color_discrete_map={'PYME': '#00CC96', 'NO PYME': '#EF553B'}, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_hist = px.histogram(df_f[df_f['VALUE_EURO'] < 1000000], x="VALUE_EURO", color="Es_PYME", title="Distribución de Precios", color_discrete_map={'PYME': '#00CC96', 'NO PYME': '#EF553B'})
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.error("Archivo 'export_CAN_2023.csv' no encontrado.")

# ==============================================================================
# SECCIÓN 3: AUDITORÍA (TU CÓDIGO ORIGINAL)
# ==============================================================================
elif seccion == "⚙️ Auditoría Técnica":
    st.title("⚙️ Auditoría Técnica del Modelo")
    st.markdown("Transparencia algorítmica y métricas de rendimiento.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Métricas de Rendimiento")
        dt = {'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'], 'Valor': ['78.5%', '72.1%', '81.4%', '76.5%']}
        st.table(pd.DataFrame(dt))
        st.info("Priorizamos el Recall para detectar PYMES reales.")
    with col2:
        st.subheader("Variables más Influyentes")
        st.bar_chart({'Historial Empresa': 0.35, 'Ratio Precio': 0.25, 'Competencia CPV': 0.15, 'País': 0.10, 'Tipo Entidad': 0.05})
