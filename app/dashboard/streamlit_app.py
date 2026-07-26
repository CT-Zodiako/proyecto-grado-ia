import sys
import streamlit as st
import pandas as pd
import json
import requests
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Asegura que el módulo hermano `diagnostics.py` sea importable sin importar
# cómo se invoque Streamlit (streamlit run agrega el directorio del script a
# sys.path, no necesariamente la raíz del repo).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostics
import similar_programs

# Configuración de la página
st.set_page_config(
    page_title="Medicina Saber Pro - Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL base de la API
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Caché para cargar datos
@st.cache_data
def load_recommendations():
    """Carga las recomendaciones desde el CSV local."""
    rec_path = Path("artifacts/recomendaciones_programa.csv")
    if rec_path.exists():
        return pd.read_csv(rec_path, encoding="utf-8")
    return None

@st.cache_data
def load_validation_results():
    """Carga los resultados de validación."""
    val_path = Path("artifacts/validation_results.json")
    if val_path.exists():
        return json.loads(val_path.read_text(encoding="utf-8"))
    return None

@st.cache_data
def load_metrics():
    """Carga las métricas del modelo."""
    met_path = Path("artifacts/metrics.json")
    if met_path.exists():
        return json.loads(met_path.read_text(encoding="utf-8"))
    return None

@st.cache_data
def load_feature_schema():
    """Carga el schema de features."""
    schema_path = Path("artifacts/feature_schema.json")
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
    return None

@st.cache_data
def load_medicina_features():
    """Carga el dataset histórico con features de medicina."""
    feat_path = Path("artifacts/medicina_features.csv")
    if feat_path.exists():
        return pd.read_csv(feat_path, encoding="utf-8")
    return None

@st.cache_data
def load_medicina_features_2025():
    """Carga el dataset extendido con datos de 2025."""
    feat_path = Path("artifacts/medicina_features_2020_2025.csv")
    if feat_path.exists():
        return pd.read_csv(feat_path, encoding="utf-8")
    # Fallback al histórico original si no existe el extendido
    return load_medicina_features()

# Funciones de API
def api_health():
    """Verifica el estado de la API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None

def api_metadata():
    """Obtiene metadatos del modelo."""
    try:
        response = requests.get(f"{API_BASE_URL}/metadata", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None

def api_predict(data):
    """Realiza una predicción."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=10)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None

# Sidebar
st.sidebar.title("🩺 Medicina Saber Pro")
st.sidebar.markdown("---")

# Configuración de API en sidebar
api_url_input = st.sidebar.text_input("URL de la API", value=API_BASE_URL)
if api_url_input != API_BASE_URL:
    API_BASE_URL = api_url_input

# Verificar estado de la API
health_data = api_health()
metadata_data = api_metadata()

if health_data:
    st.sidebar.success(f"✅ API conectada: {health_data.get('modelo_nombre', 'Desconocido')}")
else:
    st.sidebar.warning("⚠️ API no disponible. Algunas funciones estarán limitadas.")

st.sidebar.markdown("---")

# Navegación
page = st.sidebar.radio(
    "Navegación",
    ["📊 Overview", "📈 EDA", "🔮 Predicción", "🩺 Diagnóstico", "📋 Recomendaciones", "✅ Validación", "🤖 Modelos", "🔍 Explicabilidad"]
)

# ============================================
# PÁGINA: OVERVIEW
# ============================================
if page == "📊 Overview":
    st.title("Modelo de IA para Programas de Medicina en Saber Pro")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Período", "2020-2025")
    with col2:
        st.metric("Programas", "75")
    with col3:
        if health_data:
            st.metric("Modelo", health_data.get('modelo_nombre', 'N/A'))
        else:
            st.metric("Modelo", "No disponible")
    
    st.markdown("---")
    
    st.markdown("""
    ### Sobre este proyecto
    
    Este dashboard presenta el análisis y predicción del desempeño de programas de **Medicina** 
    en las pruebas **Saber Pro** de Colombia, utilizando técnicas de Machine Learning e 
    Inteligencia Artificial Explicable (XAI).
    
    ### Modelos probados
    
    Se entrenaron y compararon **9 modelos** diferentes. El seleccionado final es **Ridge**.
    
    | Modelo | Tipo |
    |---|---|
    | **Ridge** | Regresión lineal regularizada |
    | Lasso | Regresión lineal regularizada |
    | ElasticNet | Regresión lineal regularizada |
    | XGBoost | Gradient boosting |
    | LightGBM | Gradient boosting |
    | CatBoost | Gradient boosting |
    | HistGradientBoosting | Gradient boosting |
    | Random Forest | Ensemble de árboles |
    | KNN | Basado en instancias |
    
    ### Arquitectura
    - **API**: FastAPI con endpoints `/health`, `/metadata`, `/predict`
    - **Dashboard**: Streamlit con visualizaciones interactivas
    - **Modelo activo**: **Ridge v2** (regresión lineal regularizada, entrenada con datos 2020-2025 y nuevas variables históricas)
    
    ### Datos
    - Fuente: ICFES Saber Pro (2020-2025)
    - Filtro: NBC = MEDICINA
    - Variable objetivo: PROMEDIO_GLOBAL
    """)
    
    if metadata_data:
        st.markdown("---")
        st.subheader("📋 Metadatos del Modelo")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Variables Numéricas:**")
            for feat in metadata_data.get('numeric_features', []):
                st.markdown(f"- `{feat}`")
        with col2:
            st.markdown("**Variables Categóricas:**")
            for feat in metadata_data.get('categorical_features', []):
                st.markdown(f"- `{feat}`")
    
    st.markdown("---")
    st.info("💡 Usa el menú lateral para navegar entre las secciones del dashboard.")

# ============================================
# PÁGINA: EDA
# ============================================
elif page == "📈 EDA":
    st.title("Análisis Exploratorio de Datos")
    
    rec_df = load_recommendations()
    
    if rec_df is not None:
        st.success(f"📁 Datos cargados: {len(rec_df)} recomendaciones")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            region_filter = st.multiselect(
                "Región",
                options=sorted(rec_df['NOMBRE_REGION'].dropna().unique())
            )
        with col2:
            dept_filter = st.multiselect(
                "Departamento",
                options=sorted(rec_df['NOMBRE_DEPARTAMENTO'].dropna().unique())
            )
        with col3:
            cat_filter = st.multiselect(
                "Categoría",
                options=sorted(rec_df['categoria_recomendacion'].dropna().unique())
            )
        
        # Aplicar filtros
        filtered_df = rec_df.copy()
        if region_filter:
            filtered_df = filtered_df[filtered_df['NOMBRE_REGION'].isin(region_filter)]
        if dept_filter:
            filtered_df = filtered_df[filtered_df['NOMBRE_DEPARTAMENTO'].isin(dept_filter)]
        if cat_filter:
            filtered_df = filtered_df[filtered_df['categoria_recomendacion'].isin(cat_filter)]
        
        st.markdown(f"**Mostrando:** {len(filtered_df)} registros")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución de PROMEDIO_GLOBAL")
            fig = px.histogram(
                filtered_df,
                x='promedio_global_anual',
                nbins=20,
                title="Histograma de Promedio Global",
                labels={'promedio_global_anual': 'PROMEDIO_GLOBAL'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribución por Región")
            region_data = filtered_df.groupby('NOMBRE_REGION')['promedio_global_anual'].mean().reset_index()
            fig = px.bar(
                region_data,
                x='NOMBRE_REGION',
                y='promedio_global_anual',
                title="Promedio Global por Región",
                labels={'promedio_global_anual': 'Promedio', 'NOMBRE_REGION': 'Región'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top instituciones
        st.subheader("Top Instituciones por Desempeño")
        top_inst = filtered_df.groupby('NOMBRE_INSTITUCION')['promedio_global_anual'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=top_inst.values,
            y=top_inst.index,
            orientation='h',
            title="Top 10 Instituciones",
            labels={'x': 'Promedio Global', 'y': 'Institución'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de datos
        st.subheader("Datos Filtrados")
        display_cols = ['AÑO', 'NOMBRE_REGION', 'NOMBRE_DEPARTAMENTO', 'NOMBRE_INSTITUCION', 
                       'NOMBRE_PROGRAMA_ACAD', 'promedio_global_anual', 'categoria_recomendacion']
        st.dataframe(filtered_df[display_cols], use_container_width=True)
    else:
        st.error("❌ No se encontraron datos de recomendaciones. Verifica que `artifacts/recomendaciones_programa.csv` exista.")

# ============================================
# PÁGINA: PREDICCIÓN
# ============================================
elif page == "🔮 Predicción":
    st.title("Predicción de Desempeño")
    st.markdown("Seleccioná tu institución y el año que querés predecir. El sistema buscará automáticamente el historial.")
    
    schema = load_feature_schema()
    features_df = load_medicina_features_2025()
    
    if schema and features_df is not None:
        # Determinar año máximo disponible
        max_year = int(features_df['AÑO'].max())
        min_year = int(features_df['AÑO'].min())
        
        # Paso 1: Seleccionar institución
        st.subheader("1. ¿De qué programa querés predecir?")
        
        # Crear etiqueta amigable para cada programa
        features_df['etiqueta_programa'] = (
            features_df['NOMBRE_INSTITUCION'] + " — " +
            features_df['NOMBRE_MUNICIPIO'] + ", " +
            features_df['NOMBRE_DEPARTAMENTO']
        )
        
        # Algunas instituciones tienen múltiples ID_PROGRAMA_ACAD para el mismo nombre.
        # Para no confundir al usuario, agrupamos por nombre y elegimos el ID con más historia.
        program_counts = features_df.groupby(['etiqueta_programa', 'ID_INSTITUCION', 'ID_PROGRAMA_ACAD']).size().reset_index(name='n_observaciones')
        opciones = program_counts.sort_values('n_observaciones', ascending=False).drop_duplicates('etiqueta_programa')
        opciones = opciones.sort_values('etiqueta_programa').reset_index(drop=True)
        
        etiqueta_seleccionada = st.selectbox(
            "Programa de Medicina",
            options=opciones['etiqueta_programa'].tolist()
        )
        
        # Recuperar IDs del programa seleccionado (el que tiene más historia)
        seleccion = opciones[opciones['etiqueta_programa'] == etiqueta_seleccionada].iloc[0]
        id_institucion = seleccion['ID_INSTITUCION']
        id_programa = seleccion['ID_PROGRAMA_ACAD']
        
        # Historial del programa
        historial = features_df[
            (features_df['ID_INSTITUCION'] == id_institucion) &
            (features_df['ID_PROGRAMA_ACAD'] == id_programa)
        ].sort_values('AÑO').copy()
        
        # Paso 2: Seleccionar año a predecir
        st.subheader("2. ¿Para qué año querés predecir?")
        
        col1, col2 = st.columns(2)
        with col1:
            año_prediccion = st.number_input(
                "Año a predecir",
                min_value=min_year,
                max_value=2030,
                value=max_year + 1,
                step=1,
                help="Podés predecir años futuros o revisar años pasados que ya tengan datos reales."
            )
        with col2:
            st.metric("Último año con datos", str(max_year))
        
        # Paso 3: Calcular inputs automáticamente
        st.subheader("3. Datos que usará el modelo")
        
        # Buscar el último año anterior al año de predicción
        historial_previo = historial[historial['AÑO'] < año_prediccion].copy()
        
        if len(historial_previo) == 0:
            st.warning("⚠️ Este programa no tiene historia previa al año seleccionado. La predicción será menos confiable.")
            ultimo_año = None
            promedio_anterior = None
            promedio_movil = None
            desviacion_hist = None
            anios_hist = 0
        else:
            ultimo_año = historial_previo['AÑO'].max()
            ultimo_registro = historial_previo[historial_previo['AÑO'] == ultimo_año].iloc[0]
            
            # promedio_global_anterior = promedio_global_anual del último año previo
            promedio_anterior = float(ultimo_registro['promedio_global_anual'])
            
            # promedio_movil_2_anios = promedio de los últimos 2 años previos
            ultimos_2 = historial_previo.tail(2)
            promedio_movil = float(ultimos_2['promedio_global_anual'].mean())
            
            # desviacion_historica_2_anios = desviación de los últimos 2 años previos
            desviacion_hist = float(ultimos_2['promedio_global_anual'].std()) if len(ultimos_2) >= 2 else None
            
            # anios_historicos_disponibles = años de historia previa acumulada del último registro
            anios_hist = int(ultimo_registro['anios_historicos_disponibles'])
            
            # Variables categóricas del último registro
            region = str(ultimo_registro['NOMBRE_REGION'])
            departamento = str(ultimo_registro['NOMBRE_DEPARTAMENTO'])
            municipio = str(ultimo_registro['NOMBRE_MUNICIPIO'])
            institucion = str(ultimo_registro['NOMBRE_INSTITUCION'])
            programa = str(ultimo_registro['NOMBRE_PROGRAMA_ACAD'])
            
            # Mostrar resumen histórico amigable
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Último año con datos", str(int(ultimo_año)))
            with col2:
                st.metric("Promedio del último año", f"{promedio_anterior:.1f}")
            with col3:
                st.metric("Promedio móvil 2 años", f"{promedio_movil:.1f}")
            with col4:
                st.metric("Años de historia", str(anios_hist))
            
            # Mostrar tendencia histórica
            st.markdown("**Tendencia histórica del programa:**")
            fig_trend = px.line(
                historial_previo,
                x='AÑO',
                y='promedio_global_anual',
                markers=True,
                title=f"Evolución de {institucion}",
                labels={'AÑO': 'Año', 'promedio_global_anual': 'PROMEDIO_GLOBAL'}
            )
            fig_trend.update_layout(yaxis_range=[100, 200])
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Mostrar detalles técnicos en expansor (para usuarios avanzados)
            with st.expander("Ver detalles técnicos de los datos usados"):
                st.json({
                    "AÑO": año_prediccion,
                    "promedio_global_anterior": promedio_anterior,
                    "promedio_movil_2_anios": promedio_movil,
                    "desviacion_historica_2_anios": desviacion_hist,
                    "anios_historicos_disponibles": anios_hist,
                    "NOMBRE_REGION": region,
                    "NOMBRE_DEPARTAMENTO": departamento,
                    "NOMBRE_MUNICIPIO": municipio,
                    "NOMBRE_INSTITUCION": institucion,
                    "NOMBRE_PROGRAMA_ACAD": programa
                })
        
        # Botón de predicción
        st.markdown("---")
        if st.button("🔮 Predecir", type="primary"):
            if not health_data:
                st.error("❌ La API no está disponible. No se puede realizar la predicción.")
            else:
                # Nuevas features del modelo v2: usar valores del último registro histórico
                new_features = {
                    'promedio_movil_3_anios': 'promedio_movil_3_anios',
                    'desviacion_historica_3_anios': 'desviacion_historica_3_anios',
                    'tasa_crecimiento_anual': 'tasa_crecimiento_anual',
                    'maximo_historico': 'maximo_historico',
                    'minimo_historico': 'minimo_historico',
                    'diferencia_maximo_historico': 'diferencia_maximo_historico',
                    'anios_desde_inicio': 'anios_desde_inicio',
                    'ranking_departamento': 'ranking_departamento'
                }
                extras = {}
                if len(historial_previo) > 0:
                    for key, col in new_features.items():
                        if col in ultimo_registro:
                            val = ultimo_registro[col]
                            if pd.notna(val):
                                extras[key] = float(val) if isinstance(val, (int, float)) else int(val)
                            else:
                                extras[key] = None
                
                input_data = {
                    "AÑO": float(año_prediccion),
                    "promedio_global_anterior": promedio_anterior,
                    "promedio_movil_2_anios": promedio_movil,
                    "desviacion_historica_2_anios": desviacion_hist,
                    "anios_historicos_disponibles": float(anios_hist),
                    "NOMBRE_REGION": region,
                    "NOMBRE_DEPARTAMENTO": departamento,
                    "NOMBRE_MUNICIPIO": municipio,
                    "NOMBRE_INSTITUCION": institucion,
                    "NOMBRE_PROGRAMA_ACAD": programa,
                    **extras
                }
                
                result = api_predict(input_data)
                
                if result:
                    pred = result['prediccion']
                    
                    st.markdown("---")
                    st.subheader("📊 Resultado de la Predicción")
                    
                    # Interpretación amigable
                    if pred < 151:
                        interpretacion = "🔴 Bajo desempeño esperado"
                        color = "red"
                        mensaje = "El puntaje predicho está por debajo del umbral de desempeño bajo. Se recomienda revisar estrategias académicas."
                    elif pred < 172:
                        interpretacion = "🟡 Desempeño medio esperado"
                        color = "yellow"
                        mensaje = "El puntaje predicho está en el rango medio. Hay espacio para mejorar."
                    else:
                        interpretacion = "🟢 Alto desempeño esperado"
                        color = "green"
                        mensaje = "El puntaje predicho está en el rango alto."
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Puntaje predicho", f"{pred:.2f}")
                    with col2:
                        st.metric("Interpretación", interpretacion)
                    
                    st.markdown(f"**{mensaje}**")
                    
                    # Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pred,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "PROMEDIO_GLOBAL Predicho"},
                        gauge={
                            'axis': {'range': [100, 200]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [100, 151], 'color': "lightcoral"},
                                {'range': [151, 172], 'color': "lightyellow"},
                                {'range': [172, 200], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 151
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Si el año ya tiene datos reales, comparar
                    real_row = historial[historial['AÑO'] == año_prediccion]
                    if len(real_row) > 0:
                        real_value = float(real_row.iloc[0]['promedio_global_anual'])
                        error = abs(pred - real_value)
                        st.markdown("---")
                        st.subheader("📈 Comparación con valor real")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicción", f"{pred:.2f}")
                        with col2:
                            st.metric("Real", f"{real_value:.2f}")
                        with col3:
                            st.metric("Error absoluto", f"{error:.2f}")
                        
                        if error <= 6:
                            st.success(f"✅ El error es pequeño ({error:.2f} puntos), similar al MAE esperado del modelo.")
                        else:
                            st.warning(f"⚠️ El error es de {error:.2f} puntos. El modelo se equivocó más de lo habitual en este caso.")
                    else:
                        st.info("💡 El año seleccionado no tiene datos reales aún. La predicción es una estimación.")
                else:
                    st.error("❌ Error al realizar la predicción. Verifica que la API esté funcionando.")
    else:
        st.error("❌ No se encontraron datos históricos o el schema de features. Verifica que los archivos existan.")

# ============================================
# PÁGINA: DIAGNÓSTICO IA
# ============================================
elif page == "🩺 Diagnóstico":
    st.title("Diagnóstico del Programa")
    st.markdown(
        "Seleccioná un programa de Medicina y te mostramos cómo le fue, "
        "por qué le fue así, y qué tan confiable es esa lectura."
    )

    @st.cache_resource
    def load_diagnostic_bundle_cached():
        """Carga una sola vez por sesión el bundle de diagnóstico
        (coeficientes Lasso + estadísticas del scaler) desde artifacts/."""
        return diagnostics.load_diagnostic_model_bundle(Path("artifacts"))

    features_df = load_medicina_features_2025()

    if features_df is not None:
        # Reutiliza el mismo patrón de selector que la página Predicción:
        # una etiqueta amigable por programa, deduplicada por el ID con más historia.
        st.subheader("1. ¿Qué programa querés diagnosticar?")

        features_df['etiqueta_programa'] = (
            features_df['NOMBRE_INSTITUCION'] + " — " +
            features_df['NOMBRE_MUNICIPIO'] + ", " +
            features_df['NOMBRE_DEPARTAMENTO']
        )

        program_counts = features_df.groupby(['etiqueta_programa', 'ID_INSTITUCION', 'ID_PROGRAMA_ACAD']).size().reset_index(name='n_observaciones')
        opciones = program_counts.sort_values('n_observaciones', ascending=False).drop_duplicates('etiqueta_programa')
        opciones = opciones.sort_values('etiqueta_programa').reset_index(drop=True)

        etiqueta_diagnostico = st.selectbox(
            "Programa de Medicina",
            options=opciones['etiqueta_programa'].tolist(),
            key="diagnostico_selectbox"
        )

        seleccion = opciones[opciones['etiqueta_programa'] == etiqueta_diagnostico].iloc[0]
        id_institucion = seleccion['ID_INSTITUCION']
        id_programa = seleccion['ID_PROGRAMA_ACAD']

        historial = features_df[
            (features_df['ID_INSTITUCION'] == id_institucion) &
            (features_df['ID_PROGRAMA_ACAD'] == id_programa)
        ].sort_values('AÑO').copy()

        ultimo_registro = historial.iloc[-1]

        # Dos de las 4 variables del Lasso (maximo_historico,
        # promedio_movil_3_anios) no están persistidas en ningún CSV del
        # dashboard: se calculan solo en memoria durante el entrenamiento
        # (ver mejorar_modelo.py::add_new_features). Las recalculamos acá
        # mismo a partir de la serie histórica ya cargada, replicando
        # exactamente esa lógica (shift(1) + expanding/rolling), en vez de
        # asumir que existen como columnas.
        #
        # anios_hist se deriva de esta MISMA serie filtrada (en vez de leer
        # la columna persistida anios_historicos_disponibles) para que haya
        # una única fuente de verdad: si hubiera una discrepancia entre el
        # agrupamiento local (por ID_INSTITUCION+ID_PROGRAMA_ACAD) y el valor
        # persistido, usar dos fuentes distintas podría dejar anios_hist>=1
        # mientras años_previos quedó vacío, causando un TypeError en
        # compute_feature_contributions (None - media). Con una sola fuente
        # esa divergencia es estructuralmente imposible.
        años_previos = historial.iloc[:-1]
        anios_hist = len(años_previos)
        if anios_hist > 0:
            maximo_historico_valor = float(años_previos['promedio_global_anual'].max())
            promedio_movil_3_anios_valor = float(años_previos['promedio_global_anual'].tail(3).mean())
        else:
            maximo_historico_valor = None
            promedio_movil_3_anios_valor = None

        st.markdown("---")
        st.subheader("2. Cómo le fue")

        ultimo_año = int(ultimo_registro['AÑO'])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Último año disponible", str(ultimo_año))
        with col2:
            st.metric("Promedio de ese año", f"{ultimo_registro['promedio_global_anual']:.2f}")

        if len(historial) > 1:
            fig_trend = px.line(
                historial,
                x='AÑO',
                y='promedio_global_anual',
                markers=True,
                title=f"Evolución de {ultimo_registro['NOMBRE_INSTITUCION']}",
                labels={'AÑO': 'Año', 'promedio_global_anual': 'PROMEDIO_GLOBAL'}
            )
            fig_trend.update_layout(yaxis_range=[100, 200])
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")
        st.subheader("3. Por qué")

        bundle = load_diagnostic_bundle_cached()
        row = {
            "promedio_global_anterior": float(ultimo_registro["promedio_global_anterior"]),
            "promedio_movil_2_anios": float(ultimo_registro["promedio_movil_2_anios"]),
            "maximo_historico": maximo_historico_valor,
            "promedio_movil_3_anios": promedio_movil_3_anios_valor,
        }
        narrativa = diagnostics.build_diagnostic_narrative(bundle, row, anios_hist)

        if narrativa["confidence_state"] == diagnostics.ConfidenceState.ZERO:
            st.info(
                "ℹ️ **Sin diagnóstico disponible.** Este programa todavía no tiene un "
                "año anterior con el cual comparar, así que no hay una explicación "
                "de «por qué» para mostrar todavía."
            )
        else:
            st.markdown(f"**{narrativa['narrative_sentence']}**")
            with st.expander("Ver análisis de contribución de variables"):
                contrib_df = pd.DataFrame(narrativa["contributions"])[
                    ["feature", "raw_value", "contribution"]
                ]
                st.dataframe(contrib_df, use_container_width=True)
                fig_contrib = px.bar(
                    contrib_df,
                    x="contribution",
                    y="feature",
                    orientation="h",
                    title="Contribución de cada variable a la predicción"
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

        st.markdown("---")
        st.subheader("4. Qué tan confiable es esto")

        if narrativa["confidence_state"] == diagnostics.ConfidenceState.LOW:
            st.warning(
                "⚠️ **Confianza reducida.** Este programa tiene poca historia previa "
                "(menos de 2 años), así que la estimación es menos confiable que para "
                "programas con más años de datos."
            )

        metrics_data = load_metrics()
        if metrics_data:
            best_metrics = metrics_data.get("best_test_metrics", {})
            st.markdown(
                "Estimación basada en el volumen histórico del programa y el error de "
                "validación global del modelo (no es un intervalo de confianza estadístico):"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE (test)", f"{best_metrics.get('MAE', 0):.2f}")
            with col2:
                st.metric("RMSE (test)", f"{best_metrics.get('RMSE', 0):.2f}")
            with col3:
                st.metric("R² (test)", f"{best_metrics.get('R2', 0):.3f}")

        st.markdown("---")
        st.subheader("5. Programas similares")
        st.markdown(
            "Programas con un perfil histórico parecido (promedio, tendencia y "
            "volatilidad) al de este programa, sin tener en cuenta su ubicación."
        )

        @st.cache_data
        def build_similar_programs_index_cached(features_df):
            """Construye el perfil histórico (promedio, tendencia,
            volatilidad) de los 73 programas y los agrupa una sola vez
            por sesión (cacheado por contenido de features_df, mismo
            patrón que load_medicina_features_2025)."""
            perfiles = []
            for (id_inst, id_prog), grupo in features_df.groupby(
                ['ID_INSTITUCION', 'ID_PROGRAMA_ACAD']
            ):
                serie = grupo.sort_values('AÑO')['promedio_global_anual'].tolist()
                perfiles.append(
                    similar_programs.compute_program_profile(
                        {"id_institucion": id_inst, "id_programa_acad": id_prog},
                        serie,
                    )
                )
            return similar_programs.build_similar_programs_index(perfiles)

        try:
            perfiles, matriz_estandarizada, _scaler, cluster_labels = (
                build_similar_programs_index_cached(features_df)
            )
        except ValueError:
            # sklearn.cluster.KMeans exige al menos N_CLUSTERS programas para
            # agrupar. Si el dataset alguna vez tiene menos de 12 programas de
            # Medicina, degradamos con un aviso en vez de romper toda la
            # página de Diagnóstico.
            perfiles, matriz_estandarizada, cluster_labels = [], None, None

        clave_seleccionada = {"id_institucion": id_institucion, "id_programa_acad": id_programa}
        indice_seleccionado = (
            next((i for i, p in enumerate(perfiles) if p["key"] == clave_seleccionada), None)
            if perfiles else None
        )

        if indice_seleccionado is not None:
            similares = similar_programs.select_similar_programs(
                indice_seleccionado, perfiles, matriz_estandarizada, cluster_labels
            )

            etiqueta_por_clave = {
                (fila['ID_INSTITUCION'], fila['ID_PROGRAMA_ACAD']): fila['etiqueta_programa']
                for _, fila in opciones.iterrows()
            }

            for par in similares:
                clave_par = (par['key']['id_institucion'], par['key']['id_programa_acad'])
                etiqueta = etiqueta_por_clave.get(clave_par, "Programa sin etiqueta disponible")
                with st.container(border=True):
                    st.markdown(f"**{etiqueta}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Promedio histórico", f"{par['avg']:.2f}")
                    with col2:
                        st.metric("Tendencia", par['trend_label'])
        else:
            st.info(
                "ℹ️ No se pudieron calcular programas similares para esta "
                "selección (datos insuficientes para el agrupamiento)."
            )
    else:
        st.error("❌ No se encontraron datos históricos. Verificá que los archivos existan.")

# ============================================
# PÁGINA: RECOMENDACIONES
# ============================================
elif page == "📋 Recomendaciones":
    st.title("Sistema de Recomendaciones")
    
    rec_df = load_recommendations()
    
    if rec_df is not None:
        st.success(f"📁 {len(rec_df)} recomendaciones cargadas")
        
        # Conteo por categoría
        st.subheader("Distribución de Categorías")
        cat_counts = rec_df['categoria_recomendacion'].value_counts().reset_index()
        cat_counts.columns = ['Categoría', 'Cantidad']
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(cat_counts, values='Cantidad', names='Categoría', title="Categorías")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(cat_counts, x='Categoría', y='Cantidad', title="Conteo por Categoría")
            st.plotly_chart(fig, use_container_width=True)
        
        # Filtros
        st.markdown("---")
        st.subheader("🔍 Filtrar Recomendaciones")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            cat_filter = st.multiselect(
                "Categoría",
                options=sorted(rec_df['categoria_recomendacion'].unique())
            )
        with col2:
            region_filter = st.multiselect(
                "Región",
                options=sorted(rec_df['NOMBRE_REGION'].dropna().unique())
            )
        with col3:
            dept_filter = st.multiselect(
                "Departamento",
                options=sorted(rec_df['NOMBRE_DEPARTAMENTO'].dropna().unique())
            )
        
        filtered = rec_df.copy()
        if cat_filter:
            filtered = filtered[filtered['categoria_recomendacion'].isin(cat_filter)]
        if region_filter:
            filtered = filtered[filtered['NOMBRE_REGION'].isin(region_filter)]
        if dept_filter:
            filtered = filtered[filtered['NOMBRE_DEPARTAMENTO'].isin(dept_filter)]
        
        st.markdown(f"**Mostrando:** {len(filtered)} recomendaciones")
        
        # Tabla
        display_cols = [
            'AÑO', 'NOMBRE_REGION', 'NOMBRE_DEPARTAMENTO', 'NOMBRE_INSTITUCION',
            'NOMBRE_PROGRAMA_ACAD', 'promedio_global_anual', 'categoria_recomendacion',
            'prediccion_modelo', 'error_absoluto_test', 'texto_recomendacion'
        ]
        st.dataframe(filtered[display_cols], use_container_width=True)
        
        # Casos de riesgo
        st.markdown("---")
        st.subheader("🚨 Top Casos de Riesgo")
        
        riesgo_cats = ['riesgo_prioritario', 'desempeno_bajo', 'tendencia_descendente']
        riesgo_df = rec_df[rec_df['categoria_recomendacion'].isin(riesgo_cats)].copy()
        
        if len(riesgo_df) > 0:
            riesgo_df = riesgo_df.sort_values('promedio_global_anual').head(10)
            
            for _, row in riesgo_df.iterrows():
                with st.expander(f"🚨 {row['NOMBRE_INSTITUCION']} - {row['NOMBRE_PROGRAMA_ACAD']} ({row['NOMBRE_DEPARTAMENTO']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Categoría:** `{row['categoria_recomendacion']}`")
                        st.markdown(f"**Año:** {row['AÑO']}")
                        st.markdown(f"**Promedio Observado:** {row['promedio_global_anual']:.2f}")
                    with col2:
                        st.markdown(f"**Predicción:** {row['prediccion_modelo']:.2f}")
                        st.markdown(f"**Error:** {row['error_absoluto_test']:.2f}")
                        st.markdown(f"**Variación Anual:** {row['variacion_anual']:.2f}")
                    st.markdown(f"**Recomendación:** {row['texto_recomendacion']}")
        else:
            st.info("No se encontraron casos de riesgo con los filtros actuales.")
    else:
        st.error("❌ No se encontraron datos de recomendaciones.")

# ============================================
# PÁGINA: VALIDACIÓN
# ============================================
elif page == "✅ Validación":
    st.title("Validación del Modelo")
    
    val_data = load_validation_results()
    metrics_data = load_metrics()
    
    if val_data:
        st.success("📁 Datos de validación cargados")
        
        # Métricas principales
        st.subheader("📊 Métricas del Modelo")
        
        best_val = val_data.get('best_validation_metrics', {})
        best_test = val_data.get('best_test_metrics', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modelo", val_data.get('best_model_by_validation_mae', 'N/A'))
        with col2:
            st.metric("Test MAE", f"{best_test.get('MAE', 'N/A')}")
        with col3:
            st.metric("Test R²", f"{best_test.get('R2', 'N/A')}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Validación MAE", f"{best_val.get('MAE', 'N/A')}")
        with col2:
            st.metric("Validación RMSE", f"{best_val.get('RMSE', 'N/A')}")
        with col3:
            st.metric("Validación R²", f"{best_val.get('R2', 'N/A')}")
        
        # Comparación de modelos
        st.markdown("---")
        st.subheader("📈 Comparación de Modelos")
        
        # Explicación de cada modelo
        with st.expander("🧠 ¿Qué hace diferente cada modelo?"):
            st.markdown("""
            | Modelo | Tipo | Cómo aprende | Fortalezas |
            |---|---|---|---|
            | **Random Forest** | Ensemble de árboles | Entrena muchos árboles independientes con muestras aleatorias y promedia sus predicciones. | Robusto, poco propenso al overfitting, buen modelo base. |
            | **XGBoost** | Gradient boosting | Entrena árboles de forma secuencial, cada nuevo árbol corrige los errores del anterior. | Muy preciso, usa regularización para evitar sobreajuste. |
            | **LightGBM** | Gradient boosting (histogram-based) | Similar a XGBoost pero crece los árboles por hojas en lugar de por niveles. | Muy rápido, eficiente con muchas categorías. |
            | **CatBoost** | Gradient boosting | Optimizado para variables categóricas, usa codificación basada en objetivo. | Bueno con categorías como instituciones y municipios. |
            | **HistGradientBoosting** | Gradient boosting (sklearn) | Versión de sklearn de boosting con histogramas. Maneja valores faltantes. | Buen balance entre velocidad y precisión. |
            | **Ridge** | Regresión lineal regularizada | Busca una línea (combinación lineal de variables) que minimice el error con penalización L2. | Simple, interpretable, funciona muy bien cuando las relaciones son aproximadamente lineales. |
            | **Lasso** | Regresión lineal regularizada | Similar a Ridge pero con penalización L1, que puede anular variables poco importantes. | Selecciona automáticamente variables relevantes. |
            | **ElasticNet** | Regresión lineal regularizada | Combina penalizaciones L1 y L2. | Balance entre Ridge y Lasso. |
            | **KNN** | Basado en instancias | Predice el promedio de los programas más similares en el espacio de variables. | Simple, no asume forma particular de los datos. |
            
            **¿Por qué ganó Ridge?**  
            Ridge encontró una relación casi lineal entre el historial de puntajes y el puntaje futuro. Al regularizar, evitó que el modelo se ajuste demasiado a ruido. A veces un modelo simple supera a modelos complejos cuando las variables son muy informativas.
            """)
        
        model_results = val_data.get('model_results', [])
        if model_results:
            # Flatten nested structure {modelo: ..., validacion: {...}, test: {...}}
            flat_rows = []
            for r in model_results:
                modelo = r.get('modelo', '')
                for split in ['validacion', 'test']:
                    if split in r:
                        flat_rows.append({
                            'modelo': modelo,
                            'split': split.replace('validacion', 'validación').capitalize(),
                            'MAE': r[split].get('MAE'),
                            'RMSE': r[split].get('RMSE'),
                            'R2': r[split].get('R2')
                        })
            comp_df = pd.DataFrame(flat_rows)
            
            fig = px.bar(
                comp_df,
                x='modelo',
                y='MAE',
                color='split',
                barmode='group',
                title="MAE por Modelo y Split",
                labels={'MAE': 'MAE', 'modelo': 'Modelo'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.info("No se encontraron resultados de comparación de modelos.")
        
        # Residuos
        st.markdown("---")
        st.subheader("📉 Resumen de Residuos")
        
        residual = val_data.get('residual_summary', {})
        if residual:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Residuo Medio Validación", f"{residual.get('validation_residual_mean', 'N/A')}")
            with col2:
                st.metric("Residuo Medio Test", f"{residual.get('test_residual_mean', 'N/A')}")
        
        # Generalización
        gen = val_data.get('generalization_check', {})
        if gen:
            st.markdown("---")
            st.subheader("🔄 Generalización")
            st.json(gen)
    else:
        st.error("❌ No se encontraron datos de validación.")


# ============================================
# PÁGINA: MODELOS
# ============================================
elif page == "🤖 Modelos":
    st.title("Modelos Entrenados")
    st.markdown("Esta página muestra todos los modelos que se entrenaron y compararon para seleccionar el mejor.")
    
    metrics_data = load_metrics()
    
    if metrics_data:
        st.success("📁 Métricas de modelos cargadas")
        
        # Métricas del mejor modelo
        best_name = metrics_data.get('best_model_by_validation_mae', 'N/A')
        best_val = metrics_data.get('best_validation_metrics', {})
        best_test = metrics_data.get('best_test_metrics', {})
        
        st.subheader("🏆 Modelo seleccionado")
        st.info(f"**{best_name}** fue seleccionado por tener el menor MAE en validación.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valid MAE", f"{best_val.get('MAE', 'N/A'):.3f}")
        with col2:
            st.metric("Test MAE", f"{best_test.get('MAE', 'N/A'):.3f}")
        with col3:
            st.metric("Test R²", f"{best_test.get('R2', 'N/A'):.3f}")
        
        # Tabla comparativa completa
        st.markdown("---")
        st.subheader("📊 Comparación completa")
        
        model_results = metrics_data.get('model_results', [])
        if model_results:
            # Flatten nested structure {modelo: ..., validacion: {...}, test: {...}}
            flat_rows = []
            for r in model_results:
                modelo = r.get('modelo', '')
                for split in ['validacion', 'test']:
                    if split in r:
                        flat_rows.append({
                            'modelo': modelo,
                            'split': split.replace('validacion', 'validación').capitalize(),
                            'MAE': r[split].get('MAE'),
                            'RMSE': r[split].get('RMSE'),
                            'R2': r[split].get('R2')
                        })
            comp_df = pd.DataFrame(flat_rows)
            
            # Reordenar columnas para mejor lectura
            display_cols = ['modelo', 'split', 'MAE', 'RMSE', 'R2']
            comp_df = comp_df[[c for c in display_cols if c in comp_df.columns]]
            comp_df = comp_df.sort_values(['split', 'MAE'])
            
            st.dataframe(comp_df, use_container_width=True)
            
            # Gráfico de MAE por modelo
            fig = px.bar(
                comp_df,
                x='modelo',
                y='MAE',
                color='split',
                barmode='group',
                title="MAE por Modelo y Split",
                labels={'MAE': 'MAE', 'modelo': 'Modelo'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de R² por modelo
            fig2 = px.bar(
                comp_df,
                x='modelo',
                y='R2',
                color='split',
                barmode='group',
                title="R² por Modelo y Split",
                labels={'R2': 'R²', 'modelo': 'Modelo'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Explicación de modelos
        st.markdown("---")
        st.subheader("🧠 ¿Qué hace cada modelo?")
        
        st.markdown("""
        | Modelo | Tipo | Cómo aprende |
        |---|---|---|
        | **Ridge** | Regresión lineal regularizada | Busca una combinación lineal de las variables con penalización L2. |
        | **Lasso** | Regresión lineal regularizada | Similar a Ridge pero con penalización L1 que puede eliminar variables. |
        | **ElasticNet** | Regresión lineal regularizada | Combina L1 y L2. |
        | **XGBoost** | Gradient boosting | Entrena árboles secuencialmente corrigiendo errores. |
        | **LightGBM** | Gradient boosting | Similar a XGBoost pero crece árboles por hojas. |
        | **CatBoost** | Gradient boosting | Optimizado para variables categóricas. |
        | **HistGradientBoosting** | Gradient boosting | Versión de sklearn con histogramas. |
        | **Random Forest** | Ensemble de árboles | Promedio de muchos árboles independientes. |
        | **KNN** | Basado en instancias | Predice según los programas más similares. |
        """)
        
        st.info("""
        **¿Por qué ganó Ridge?**
        
        Ridge encontró que la relación entre el historial de puntajes y el puntaje futuro es aproximadamente lineal.
        La regularización L2 evitó el sobreajuste. A veces un modelo simple supera a modelos complejos cuando
        las variables son muy informativas.
        """)
    else:
        st.error("❌ No se encontraron métricas de modelos.")


# PÁGINA: EXPLICABILIDAD
# ============================================
elif page == "🔍 Explicabilidad":
    st.title("Explicabilidad del Modelo")
    
    schema = load_feature_schema()
    val_data = load_validation_results()
    
    if schema:
        st.success("📁 Schema cargado")
        
        # Importancia de features
        st.subheader("📊 Importancia de Variables")
        
        top_features = val_data.get('explainability_top_rf_original_features', []) if val_data else []
        
        if top_features:
            feat_df = pd.DataFrame(top_features)
            feat_df = feat_df.sort_values('importance', ascending=True)
            
            fig = px.bar(
                feat_df,
                x='importance',
                y='feature_original',
                orientation='h',
                title="Importancia de Variables (Random Forest)",
                labels={'importance': 'Importancia', 'feature_original': 'Variable'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se encontraron datos de importancia de features.")
        
        # Variables excluidas
        st.markdown("---")
        st.subheader("🚫 Variables Excluidas")
        
        excluded = schema.get('excluded_features', [])
        if excluded:
            st.markdown("El modelo limpio **no usa** estas variables:")
            for feat in excluded:
                st.markdown(f"- ~~`{feat}`~~")
        
        # SHAP
        st.markdown("---")
        st.subheader("🔮 SHAP (Inteligencia Artificial Explicable)")
        
        st.info("""
        💡 **Nota sobre SHAP:**
        
        Los gráficos SHAP (Summary, Dependence, Waterfall) se generan automáticamente 
        cuando el paquete `shap` está instalado en el entorno de ejecución.
        
        En este entorno local, SHAP no está instalado, pero el notebook `modelo_medicina.ipynb` 
        incluye el código completo para generar estos gráficos cuando `shap` esté disponible.
        
        Para ver los gráficos SHAP:
        1. Instalá `shap` en Colab: `!pip install shap`
        2. Ejecutá la sección 7 del notebook
        3. Los gráficos se renderizarán automáticamente
        """)
        
        # Contrato de predicción
        st.markdown("---")
        st.subheader("📋 Contrato de Predicción")
        
        st.markdown("**Variables requeridas:**")
        for feat in schema.get('numeric_features', []):
            st.markdown(f"- `{feat}` (numérica)")
        for feat in schema.get('categorical_features', []):
            st.markdown(f"- `{feat}` (categórica)")
        
        st.markdown("**Variable objetivo:**")
        st.markdown(f"- `{schema.get('target', 'promedio_global_anual')}`")
    else:
        st.error("❌ No se encontró el schema de features.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Medicina Saber Pro Dashboard**  
Proyecto de Especialización en IA  
Periodo: 2020-2025
""")
