import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os
import urllib.request

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
# 2. FUNCIONES MATEMÁTICAS FUZZY
# ==============================================================================
def membership_trapezoidal(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if a < x < b: return (x - a) / (b - a)
    if b <= x <= c: return 1.0
    if c < x < d: return (d - x) / (d - c)
    return 0.0

def membership_triangular(x, a, b, c):
    return max(min((x - a) / (b - a), (c - x) / (c - b)), 0) if b != a and c != b else 0.0

# ==============================================================================
# 3. CARGA DEL CEREBRO (OPTIMIZADO PARA GOOGLE DRIVE Y RENDER)
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
    # INSTRUCCIÓN: Reemplaza 'TU_ID_AQUÍ' con el ID que obtuviste de Google Drive
    ID_ARCHIVO_DRIVE = "1jOCGQTRZfNNoF1kGHD_S6OAxgUkLmC6c"
    URL_DESCARGA = f"https://drive.google.com/uc?export=download&id={ID_ARCHIVO_DRIVE}"
    NOMBRE_LOCAL = "datos_tesis.joblib"
    
    try:
        # Si el archivo no está en el servidor de Render, se descarga de Drive
        if not os.path.exists(NOMBRE_LOCAL):
            with st.spinner('Conectando con Google Drive para cargar base de datos (50MB)...'):
                urllib.request.urlretrieve(URL_DESCARGA, NOMBRE_LOCAL)
        
        return joblib.load(NOMBRE_LOCAL)
    except Exception as e:
        # Intento de carga local por si acaso el archivo ya estuviera ahí
        try:
            return joblib.load(NOMBRE_LOCAL)
        except:
            st.error(f"🚨 Error crítico: No se pudo cargar el modelo desde Drive ni localmente. {e}")
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
# 4. INTERFAZ GRÁFICA (PANEL IZQUIERDO)
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

    def activar_analisis():
        st.session_state['analisis_realizado'] = True

    st.markdown("---")
    btn_calcular = st.button("🚀 Calcular Viabilidad", type="primary", use_container_width=True, on_click=activar_analisis)

# ==============================================================================
# 5. MOTOR DE CÁLCULO Y RESULTADOS (HÍBRIDO: ML + FUZZY + REGLAS MANUALES)
# ==============================================================================
if st.session_state['analisis_realizado']:
    with col_result:
        st.subheader("2. Resultados del Análisis")
        
        if 'resultado_base' not in st.session_state:
            with st.spinner('Procesando datos y aplicando lógica experta...'):
                # 5.1 BÚSQUEDA DE DATOS CPV
                cpv_input = cpv_code.strip()
                promedio_precio_sector = None
                competencia_media_grafico = None
                competencia_total_modelo = None 
                posibles_claves = [cpv_input, int(cpv_input) if cpv_input.isdigit() else None]
                
                for k in posibles_claves:
                    if k in ref_promedio_precio: promedio_precio_sector = ref_promedio_precio[k]; break
                for k in posibles_claves:
                    if k in ref_promedio_competidores: competencia_media_grafico = ref_promedio_competidores[k]; break
                for k in posibles_claves:
                    if k in ref_total_licitaciones: competencia_total_modelo = ref_total_licitaciones[k]; break

                if promedio_precio_sector is None: promedio_precio_sector = valor_euro 
                if competencia_media_grafico is None: competencia_media_grafico = 5.0 
                if competencia_total_modelo is None: competencia_total_modelo = 10 

                historia = ref_participacion.get(empresa, 0)
                ratio_valor = valor_euro / (promedio_precio_sector if promedio_precio_sector != 0 else 1)

                # 5.2 MACHINE LEARNING
                input_df = pd.DataFrame({
                    'Valor_Estimado_EUR': [valor_euro], 'Num_Ofertas_Recibidas': [num_ofertas],
                    'Participacion_Historica_Empresa': [historia], 'Competencia_Sector_CPV': [competencia_total_modelo],
                    'Ratio_Valor_Sector': [ratio_valor], 'Codigo_CPV_Sector': [cpv_code], 'ISO_COUNTRY_CODE': [pais],
                    'TYPE_OF_CONTRACT': [tipo_contrato], 'Tipo_Procedimiento': [tipo_proc],
                    'MAIN_ACTIVITY': [actividad], 'CRIT_CODE': [criterio], 'CAE_TYPE': [tipo_entidad]
                })
                prob_ml_raw = modelo.predict_proba(input_df)[0][1]

                # 5.3 LÓGICA DE PENALIZACIÓN MANUAL (REQUISITOS DE TESIS)
                mensajes_explicativos = []
                penalizacion_total = 0.0
                
                # Regla 1: Historia (12.5%)
                if historia == 0:
                    penalizacion_total += 0.125
                    mensajes_explicativos.append("📉 **Historial:** Sin experiencia previa. Penalización: **-12.5%**.")
                
                # Regla 2: Escala de Competencia
                if num_ofertas == 2: 
                    penalizacion_total += 0.10
                    mensajes_explicativos.append("👥 **Competencia:** 2 oferentes. Penalización: **-10%**.")
                elif num_ofertas == 3: 
                    penalizacion_total += 0.20
                    mensajes_explicativos.append("👥 **Competencia:** 3 oferentes. Penalización: **-20%**.")
                elif num_ofertas >= 5: 
                    penalizacion_total += 0.25
                    mensajes_explicativos.append("⚠️ **Saturación:** >= 5 oferentes. Penalización: **-25%**.")

                # Regla 3: Precio Alto (Fuzzy)
                if valor_euro > (promedio_precio_sector * 1.15):
                    penalizacion_total += 0.15
                    mensajes_explicativos.append("💰 **Precio:** Oferta superior al mercado. Penalización: **-15%**.")

                prob_final = max(0.01, min(0.99, prob_ml_raw - penalizacion_total))
                
                st.session_state['resultado_base'] = prob_final
                st.session_state['mensajes_base'] = mensajes_explicativos
                st.session_state['metricas_base'] = {
                    'historia': historia, 'ratio': ratio_valor, 'competencia': num_ofertas,
                    'promedio_sector': promedio_precio_sector, 'penalizacion': penalizacion_total,
                    'comp_media_grafico': competencia_media_grafico
                }

        # --- VISUALIZACIÓN ---
        prob_base = st.session_state['resultado_base']
        msgs = st.session_state['mensajes_base']
        mets = st.session_state['metricas_base']

        if prob_base > 0.5:
            st.success(f"### ✅ PROBABILIDAD DE ÉXITO: {prob_base:.2%}")
        else:
            st.error(f"### ⚠️ PROBABILIDAD DE ÉXITO: {prob_base:.2%}")
        st.progress(prob_base)
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Historial", f"{mets['historia']} ganadas")
        k2.metric("Ratio Precio", f"{mets['ratio']:.2f}x")
        k3.metric("Competencia", f"{mets['competencia']} empresas")

        # --- SIMULADOR DE DESCUENTO ---
        st.markdown("---")
        st.subheader("💡 Simulador de Competitividad")
        with st.container(border=True):
            val_descuento = st.slider("Descuento a aplicar (%)", 0, 30, 0, key="slider_final")
            beneficio_pct = (val_descuento * 0.01) if val_descuento <= 20 else (0.20 + (val_descuento-20)*0.002)
            
            prob_simulada = max(0.01, min(0.99, prob_base + beneficio_pct))
            nuevo_precio_sim = valor_euro * (1 - (val_descuento/100))
            
            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Nuevo Precio", f"€ {nuevo_precio_sim:,.0f}")
            cs2.metric("Mejora", f"+{beneficio_pct*100:.1f}%")
            cs3.metric("Prob. Simulada", f"{prob_simulada:.2%}", delta=f"{(prob_simulada - prob_base):+.2%}")

        # --- GRÁFICOS ---
        st.markdown("#### 📊 Comparativa de Mercado")
        g1, g2 = st.columns(2)
        
        fig_p = go.Figure(go.Bar(
            x=['Tu Oferta', 'Media Sector'], y=[valor_euro, mets['promedio_sector']],
            marker_color=['#EF553B' if valor_euro > mets['promedio_sector'] else '#00CC96', '#636EFA'],
            text=[f"€{valor_euro:,.0f}", f"€{mets['promedio_sector']:,.0f}"], textposition='auto'
        ))
        fig_p.update_layout(height=250, margin=dict(t=20,b=0,l=0,r=0), title="Precio")
        g1.plotly_chart(fig_p, use_container_width=True)

        fig_c = go.Figure(go.Bar(
            x=['Actual', 'Histórica'], y=[num_ofertas, mets['comp_media_grafico']],
            marker_color=['#AB63FA', '#FFA15A'],
            text=[f"{num_ofertas}", f"{mets['comp_media_grafico']:.1f}"], textposition='auto'
        ))
        fig_c.update_layout(height=250, margin=dict(t=20,b=0,l=0,r=0), title="Nº Competidores")
        g2.plotly_chart(fig_c, use_container_width=True)

        with st.expander("📝 Factores de Riesgo", expanded=False):
            for m in msgs: st.markdown(f"- {m}")
            st.caption(f"Ajuste total por reglas de tesis: -{mets['penalizacion']*100:.1f}%")