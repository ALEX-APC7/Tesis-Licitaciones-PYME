import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

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
# FUNCIONES MATEMÁTICAS FUZZY
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
# 2. CARGA DE DATOS
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
    try:
        return joblib.load('datos_tesis.joblib')
    except FileNotFoundError:
        return None

sistema = cargar_cerebro()

if sistema is None:
    st.error("🚨 ERROR CRÍTICO: No se encuentra el archivo 'datos_tesis.joblib'.")
    st.stop()

modelo = sistema['modelo_entrenado']
ref_participacion = sistema['ref_participacion']
ref_promedio_precio = sistema.get('ref_promedio_sector', sistema.get('ref_promedio_precio', {}))
ref_total_licitaciones = sistema.get('ref_total_licitaciones', sistema.get('ref_competencia_cpv', {}))
ref_promedio_competidores = sistema.get('ref_promedio_competidores', {})

# ==============================================================================
# 3. INTERFAZ GRÁFICA (PANEL IZQUIERDO)
# ==============================================================================
st.title("🏛️ Sistema de Viabilidad de Licitaciones")
st.markdown("**Análisis inteligente para PYMES en el mercado europeo**")
st.markdown("---")

# --- CONTROL DE ESTADO (MEMORIA) ---
if 'analisis_realizado' not in st.session_state:
    st.session_state['analisis_realizado'] = False

def resetear_analisis():
    """Solo borra los resultados si cambias los datos de entrada."""
    st.session_state['analisis_realizado'] = False
    # Borramos el resultado base para obligar a recalcular si cambian inputs
    if 'resultado_base' in st.session_state: del st.session_state['resultado_base']

col_panel, col_result = st.columns([1, 1.5], gap="large")

with col_panel:
    st.subheader("1. Datos del Proyecto")
    
    st.markdown("##### 💶 Variable Económica")
    # Los cambios aquí SÍ resetean el análisis
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
# 4. MOTOR DE CÁLCULO Y RESULTADOS
# ==============================================================================
if st.session_state['analisis_realizado']:
    with col_result:
        st.subheader("2. Resultados del Análisis")
        
        # --- PARTE 1: CÁLCULO PESADO (SOLO SE HACE UNA VEZ) ---
        # Verificamos si ya tenemos el "resultado_base" guardado en memoria.
        # Si NO existe, lo calculamos. Si YA existe, nos saltamos este bloque.
        if 'resultado_base' not in st.session_state:
            
            with st.spinner('Procesando datos y aplicando lógica difusa...'):
                
                # 4.1 BÚSQUEDA DE DATOS
                cpv_input = cpv_code.strip()
                promedio_precio_sector = None
                competencia_media_grafico = None
                competencia_total_modelo = None 
                posibles_claves = [cpv_input]
                if cpv_input.isdigit(): posibles_claves.append(int(cpv_input))
                
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
                if promedio_precio_sector == 0: promedio_precio_sector = 1
                ratio_valor = valor_euro / promedio_precio_sector

                # 4.2 MACHINE LEARNING BASE
                input_df = pd.DataFrame({
                    'Valor_Estimado_EUR': [valor_euro], 'Num_Ofertas_Recibidas': [num_ofertas],
                    'Participacion_Historica_Empresa': [historia], 'Competencia_Sector_CPV': [competencia_total_modelo],
                    'Ratio_Valor_Sector': [ratio_valor], 'Codigo_CPV_Sector': [cpv_code], 'ISO_COUNTRY_CODE': [pais],
                    'TYPE_OF_CONTRACT': [tipo_contrato], 'Tipo_Procedimiento': [tipo_proc],
                    'MAIN_ACTIVITY': [actividad], 'CRIT_CODE': [criterio], 'CAE_TYPE': [tipo_entidad]
                })
                prob_ml_raw = modelo.predict_proba(input_df)[0][1]

                # 4.3 LÓGICA DIFUSA (FUZZY LOGIC)
                mu_hist_nula = membership_trapezoidal(historia, -1, 0, 0, 5)
                mu_comp_baja = membership_trapezoidal(num_ofertas, 0, 1, 1, 2)
                mu_comp_media = membership_triangular(num_ofertas, 1, 3, 5)
                mu_comp_alta = membership_trapezoidal(num_ofertas, 3, 5, 20, 30)

                mu_precio_riesgo = 0.0
                if valor_euro > (promedio_precio_sector * 1.1):
                    mu_precio_riesgo = min(1.0, (valor_euro - promedio_precio_sector) / promedio_precio_sector)

                # Inferencia
                mensajes_explicativos = []
                penalizacion_total = 0.0
                
                if mu_hist_nula > 0.5:
                    penalizacion_total += 0.125
                    mensajes_explicativos.append("📉 **Historial:** Al no tener adjudicaciones previas, el sistema reduce la probabilidad un **12.5%**.")
                
                factor_comp = 0.0
                if num_ofertas == 2: 
                    factor_comp = 0.10
                    mensajes_explicativos.append("👥 **Competencia:** Con 2 competidores, existe una penalización leve del **10%**.")
                elif num_ofertas == 3: 
                    factor_comp = 0.20
                    mensajes_explicativos.append("👥 **Competencia:** Con 3 competidores, la dificultad aumenta. Penalización del **20%**.")
                elif num_ofertas >= 4: 
                    factor_comp = 0.25
                    mensajes_explicativos.append("⚠️ **Saturación:** Alta competencia detectada. Penalización máxima del **25%**.")
                penalizacion_total += factor_comp

                if mu_precio_riesgo > 0.2:
                    penalizacion_total += 0.15
                    mensajes_explicativos.append("💰 **Precio:** Tu oferta está por encima del promedio del sector. Penalización del **15%**.")

                prob_final_fuzzy = max(0.01, min(0.99, prob_ml_raw - penalizacion_total))
                
                # --- GUARDADO EN MEMORIA ---
                # Aquí está la clave: Guardamos todo para usarlo después con el slider
                st.session_state['resultado_base'] = prob_final_fuzzy
                st.session_state['mensajes_base'] = mensajes_explicativos
                st.session_state['metricas_base'] = {
                    'historia': historia,
                    'ratio': ratio_valor,
                    'competencia': num_ofertas,
                    'promedio_sector': promedio_precio_sector,
                    'penalizacion': penalizacion_total
                }

        # --- PARTE 2: RECUPERACIÓN Y VISUALIZACIÓN ---
        # Recuperamos los datos de la memoria. Esto es instantáneo.
        prob_base = st.session_state['resultado_base']
        msgs = st.session_state['mensajes_base']
        mets = st.session_state['metricas_base']

        # Barra de Estado Principal
        if prob_base > 0.5:
            st.success(f"### ✅ PROBABILIDAD DE ÉXITO: {prob_base:.2%}")
        else:
            st.error(f"### ⚠️ PROBABILIDAD DE ÉXITO: {prob_base:.2%}")
        
        st.progress(prob_base)
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Historial", f"{mets['historia']} ganadas")
        k2.metric("Ratio Precio", f"{mets['ratio']:.2f}x")
        k3.metric("Competencia", f"{mets['competencia']} empresas")

        # --- PARTE 3: SIMULADOR DE DESCUENTO (TOTALMENTE LIBERADO) ---
        st.markdown("---")
        st.subheader("💡 Simulador de Competitividad")
        
        with st.container(border=True):
            st.info("Mueve la barra libremente para ajustar tu estrategia:")
            
            # Slider simple y limpio. Sin callbacks, sin UUIDs raros.
            # Al moverlo, Streamlit recarga la página, pero como 'resultado_base' ya existe,
            # NO vuelve a calcular el Machine Learning, solo ejecuta la suma de abajo.
            val_descuento = st.slider("Descuento a aplicar (%)", 0, 30, 0, key="simulador_libre")
            
            # Cálculo instantáneo
            beneficio_pct = (val_descuento * 0.012) if val_descuento <= 10 else (0.12 + (val_descuento-10)*0.005)
            
            prob_simulada = max(0.01, min(0.99, prob_base + beneficio_pct))
            nuevo_precio_sim = valor_euro * (1 - (val_descuento/100))
            
            # Métricas
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Precio Ofertado", f"€ {nuevo_precio_sim:,.0f}")
            col_s2.metric("Mejora Probabilidad", f"+{beneficio_pct*100:.1f}%")
            col_s3.metric("Nueva Probabilidad", f"{prob_simulada:.2%}", delta=f"{(prob_simulada - prob_base):+.2%}")

        # --- GRÁFICOS ---
        st.markdown("#### 📊 Comparativa de Mercado")
        try:
            prom_sec = mets['promedio_sector']
            color_p = '#00CC96' if valor_euro <= prom_sec else '#EF553B'
            fig_p = go.Figure(go.Bar(
                x=['Tu Oferta', 'Media Sector'], 
                y=[valor_euro, prom_sec], 
                marker_color=[color_p, '#636EFA'], 
                text=[f"€{valor_euro/1000:.0f}k", f"€{prom_sec/1000:.0f}k"], 
                textposition='auto'
            ))
            fig_p.update_layout(title="Tu Precio vs Mercado", height=200, margin=dict(t=30,b=0))
            st.plotly_chart(fig_p, use_container_width=True)
        except Exception:
            st.warning("Datos insuficientes para la gráfica.")

        # --- CUADRO DE LÓGICA DIFUSA ---
        st.markdown("---")
        with st.expander("📝 Factores de Riesgo Detectados", expanded=False):
            if not msgs:
                st.success("✅ Tu perfil es competitivo. No hay penalizaciones graves.")
            else:
                st.write("El sistema ajustó la probabilidad por:")
                for msg in msgs:
                    st.markdown(f"- {msg}")
            
            st.caption(f"Ajuste total aplicado por lógica difusa: -{mets['penalizacion']*100:.1f}%")