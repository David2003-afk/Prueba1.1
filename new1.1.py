import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="COVID-19 Dashboard Avanzado", layout="wide", page_icon="🦠")

# Configuración inicial
GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports"

@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
    yyyy, mm, dd = yyyy_mm_dd.split("-")
    url = f"{GITHUB_BASE}/{mm}-{dd}-{yyyy}.csv"
    try:
        df = pd.read_csv(url)
        # normalizar nombres por si varían
        lower = {c.lower(): c for c in df.columns}
        cols = {
            "country": lower.get("country_region", "Country_Region"),
            "province": lower.get("province_state", "Province_State"),
            "confirmed": lower.get("confirmed", "Confirmed"),
            "deaths": lower.get("deaths", "Deaths"),
            "recovered": lower.get("recovered", "Recovered") if "recovered" in lower else None,
            "active": lower.get("active", "Active") if "active" in lower else None,
            "lat": lower.get("latitude", "Latitude") if "latitude" in lower else None,
            "long": lower.get("longitude", "Longitude") if "longitude" in lower else None,
        }
        return df, url, cols
    except:
        st.error(f"No se pudo cargar el reporte para la fecha {yyyy_mm_dd}")
        return pd.DataFrame(), url, {}

@st.cache_data(show_spinner=False)
def load_time_series(series_type="confirmed"):
    """Cargar series temporales globales"""
    url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{series_type}_global.csv"
    df = pd.read_csv(url)
    # Reformatear a formato largo
    id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']
    df = df.melt(id_vars=id_vars, var_name='Date', value_name=series_type.capitalize())
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data(show_spinner=False)
def load_population_data():
    """Cargar datos de población por país"""
    # Datos de población aproximados (fuente: World Bank 2020 approx)
    pop_data = {
        'US': 331000000,
        'United States': 331000000,
        'India': 1380000000,
        'Brazil': 212000000,
        'Russia': 146000000,
        'United Kingdom': 68200000,
        'UK': 68200000,
        'France': 65200000,
        'Germany': 83100000,
        'Italy': 60500000,
        'Spain': 46700000,
        'Argentina': 45100000,
        'Colombia': 50800000,
        'Mexico': 128900000,
        'Peru': 32900000,
        'Chile': 19100000,
        'China': 1402000000,
        'Japan': 126000000,
        'South Korea': 51700000,
        'Korea, South': 51700000,
        'Australia': 25600000,
        'Canada': 38000000,
        'South Africa': 59300000
    }
    return pop_data

# Cargar datos de población
population_data = load_population_data()

# Sidebar con filtros
st.sidebar.title("Opciones y Filtros")

# Selector de fecha
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
df, source_url, cols = load_daily_report(fecha_str)

if df.empty:
    st.error("No hay datos disponibles para la fecha seleccionada. Por favor, elija otra fecha.")
    st.stop()

st.sidebar.caption(f"Fuente: {source_url}")

# Obtener lista real de países disponibles
paises_disponibles = sorted(df[cols['country']].unique())

# Función para encontrar nombres alternativos de países
def encontrar_paises_similares(paises_disponibles, paises_buscados):
    paises_encontrados = []
    for pais_buscado in paises_buscados:
        # Buscar coincidencias exactas primero
        if pais_buscado in paises_disponibles:
            paises_encontrados.append(pais_buscado)
        else:
            # Buscar coincidencias parciales
            for pais_disponible in paises_disponibles:
                if pais_buscado.lower() in pais_disponible.lower():
                    paises_encontrados.append(pais_disponible)
                    break
            else:
                # Si no se encuentra, usar el primer país disponible
                if paises_disponibles:
                    paises_encontrados.append(paises_disponibles[0])
    return paises_encontrados

# Filtros adicionales
st.sidebar.subheader("Filtros de Datos")

# Encontrar países equivalentes para los valores por defecto
paises_por_defecto = encontrar_paises_similares(paises_disponibles, ["US", "Brazil", "India", "Russia", "United Kingdom"])

pais_seleccionado = st.sidebar.multiselect("Países", paises_disponibles, default=paises_por_defecto)

# Filtro por número mínimo de casos confirmados
min_confirmados = st.sidebar.number_input("Umbral mínimo de confirmados", min_value=0, value=1000)
df_filtrado = df[df[cols['confirmed']] >= min_confirmados]

# Organización en pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Visión General", 
    "Estadística Avanzada", 
    "Modelado Temporal", 
    "Clustering y PCA", 
    "Calidad de Datos",
    "Propuesta de Startup"
])

# ---------------------------------------------------------------
# Pestaña 1: Visión General
# ---------------------------------------------------------------
with tab1:
    st.header("Visión General de Datos COVID-19")
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    total_confirmados = df[cols['confirmed']].sum()
    total_fallecidos = df[cols['deaths']].sum()
    tasa_mortalidad = (total_fallecidos / total_confirmados * 100) if total_confirmados > 0 else 0
    
    with col1:
        st.metric("Total Confirmados", f"{total_confirmados:,}")
    with col2:
        st.metric("Total Fallecidos", f"{total_fallecidos:,}")
    with col3:
        st.metric("Tasa de Mortalidad", f"{tasa_mortalidad:.2f}%")
    with col4:
        # Calcular tasa por 100k habitantes (aproximación)
        poblacion_estimada = 0
        for pais in paises_disponibles:
            for key in population_data:
                if key.lower() in pais.lower() or pais.lower() in key.lower():
                    poblacion_estimada += population_data[key]
                    break
        
        tasa_100k = (total_fallecidos / poblacion_estimada * 100000) if poblacion_estimada > 0 else 0
        st.metric("Tasa por 100k hab", f"{tasa_100k:.2f}")
    
    # a) Mostrar todas las filas del dataset, luego volver al estado inicial
    st.subheader("Datos Completos")
    mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
    if mostrar_todas:
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(df.head(25), use_container_width=True)
    
    # b) Mostrar todas las columnas del dataset
    st.subheader("Columnas Disponibles")
    with st.expander("Ver todas las columnas"):
        st.write(list(df.columns))
    
    # Mapa interactivo
    st.subheader("Mapa de Distribución Global")
    if cols['lat'] and cols['long'] and cols['lat'] in df.columns and cols['long'] in df.columns:
        # Preparar datos para el mapa
        map_data = df[[cols['lat'], cols['long'], cols['confirmed'], cols['deaths'], cols['country']]].copy()
        map_data.columns = ['lat', 'lon', 'confirmed', 'deaths', 'country']
        map_data = map_data.dropna(subset=['lat', 'lon'])
        
        # Crear mapa
        fig = px.scatter_mapbox(map_data, 
                               lat="lat", 
                               lon="lon", 
                               size="confirmed",
                               color="deaths",
                               hover_name="country",
                               hover_data=["confirmed", "deaths"],
                               zoom=1,
                               height=500)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Los datos de latitud/longitud no están disponibles para esta fecha.")
    
    # Gráficos de barras top-N
    st.subheader("Top Países por Métricas")
    metric_option = st.selectbox("Seleccione métrica", ["Confirmados", "Fallecidos", "Tasa de Mortalidad"])
    
    # Preparar datos agregados por país
    country_agg = df.groupby(cols['country']).agg({
        cols['confirmed']: 'sum',
        cols['deaths']: 'sum'
    }).reset_index()
    country_agg['CFR'] = (country_agg[cols['deaths']] / country_agg[cols['confirmed']]) * 100
    country_agg = country_agg[country_agg[cols['confirmed']] > 0]  # Eliminar división por cero
    
    # Ordenar según la métrica seleccionada
    if metric_option == "Confirmados":
        top_countries = country_agg.nlargest(10, cols['confirmed'])
        values = top_countries[cols['confirmed']]
    elif metric_option == "Fallecidos":
        top_countries = country_agg.nlargest(10, cols['deaths'])
        values = top_countries[cols['deaths']]
    else:
        top_countries = country_agg[country_agg['CFR'].notna()].nlargest(10, 'CFR')
        values = top_countries['CFR']
    
    # Crear gráfico de barras
    fig = px.bar(top_countries, 
                 x=cols['country'], 
                 y=values,
                 title=f"Top 10 Países por {metric_option}",
                 labels={cols['country']: 'País', 'y': metric_option})
    st.plotly_chart(fig, use_container_width=True)

    st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# c) Línea del total de fallecidos (>2500) vs Confirmed/Recovered/Active por país
st.header("c) Gráfica de líneas por país (muertes > 2500)")
C, D = cols["confirmed"], cols["deaths"]
R, A = cols["recovered"], cols["active"]

metrics = [m for m in [C, D, R, A] if m and m in df.columns]
base = df[[cols["country"]] + metrics].copy()
base = base.rename(columns={cols["country"]: "Country_Region"})

filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)

if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# d) Barras de fallecidos de estados de Estados Unidos
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = cols["country"]
prov_col = cols["province"]

dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20, key="us_slider")
    st.bar_chart(agg_us.head(top_n))

# e) Gráfica de sectores (simulada con barra si no hay pie nativo)
st.header("e) Gráfica de sectores (simulada)")
# Usar países reales disponibles en lugar de valores fijos
paises_latam_disponibles = [p for p in paises_disponibles if p in ["Colombia", "Chile", "Peru", "Argentina", "Mexico", "Brazil"]]
if not paises_latam_disponibles:
    paises_latam_disponibles = paises_disponibles[:5]  # Tomar los primeros 5 si no hay coincidencias

sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), 
                    default=paises_latam_disponibles, key="sectores")
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    # Como Streamlit no tiene pie nativo, mostramos distribución normalizada como barra
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# f) Histograma del total de fallecidos por país (simulado con bar_chart)
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# g) Boxplot de Confirmed, Deaths, Recovered, Active (simulado con box_chart)
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
# Streamlit no tiene boxplot nativo, así que mostramos estadísticas resumen en tabla
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)
# ---------------------------------------------------------------
# Pestaña 2: Estadística Avanzada
# ---------------------------------------------------------------
with tab2:
    st.header("Análisis Estadístico Avanzado")
    
    # 2.1. Calcular métricas clave por país
    st.subheader("Métricas Clave por País")
    
    # Preparar datos con todas las métricas
    country_stats = df.groupby(cols['country']).agg({
        cols['confirmed']: 'sum',
        cols['deaths']: 'sum'
    }).reset_index()
    
    # Calcular CFR (Case Fatality Rate)
    country_stats['CFR'] = (country_stats[cols['deaths']] / country_stats[cols['confirmed']]) * 100
    
    # Calcular tasas por 100k habitantes (aproximación)
    def calcular_tasa_100k(row):
        pais = row[cols['country']]
        for key in population_data:
            if key.lower() in pais.lower() or pais.lower() in key.lower():
                return (row[cols['deaths']] / population_data[key]) * 100000
        return 0
    
    country_stats['Tasa_100k'] = country_stats.apply(calcular_tasa_100k, axis=1)
    
    st.dataframe(country_stats, use_container_width=True)
    
    # 2.2. Intervalos de confianza para CFR
    st.subheader("Intervalos de Confianza para CFR")
    
    # Seleccionar país para análisis
    pais_analisis = st.selectbox("Seleccione país para análisis", country_stats[cols['country']].unique())
    
    if pais_analisis:
        pais_data = country_stats[country_stats[cols['country']] == pais_analisis].iloc[0]
        n = pais_data[cols['confirmed']]
        p = pais_data['CFR'] / 100  # Proporción
        
        # Calcular intervalo de confianza (95%)
        if n > 0 and 0 < p < 1:
            z = 1.96  # Para 95% CI
            se = np.sqrt(p * (1 - p) / n)
            lower_bound = (p - z * se) * 100
            upper_bound = (p + z * se) * 100
            
            st.write(f"**CFR para {pais_analisis}:** {pais_data['CFR']:.2f}%")
            st.write(f"**Intervalo de confianza 95%:** ({lower_bound:.2f}%, {upper_bound:.2f}%)")
            
            # Visualización del intervalo
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode = "number+delta",
                value = pais_data['CFR'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {"text": f"CFR (%) - {pais_analisis}"},
                delta = {'reference': lower_bound, 'relative': False},
                number = {'suffix': "%"}
            ))
            fig.add_shape(type="line", x0=0.1, y0=lower_bound, x1=0.9, y1=lower_bound, 
                         line=dict(color="Red", width=2, dash="dash"))
            fig.add_shape(type="line", x0=0.1, y0=upper_bound, x1=0.9, y1=upper_bound, 
                         line=dict(color="Red", width=2, dash="dash"))
            fig.add_annotation(x=0.5, y=lower_bound, text="Límite inferior 95% CI", showarrow=True, arrowhead=1)
            fig.add_annotation(x=0.5, y=upper_bound, text="Límite superior 95% CI", showarrow=True, arrowhead=1)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Datos insuficientes para calcular intervalo de confianza")
    
    # 2.3. Test de hipótesis de proporciones para comparar CFR entre dos países
    st.subheader("Comparación de CFR entre Países")
    
    col1, col2 = st.columns(2)
    with col1:
        pais1 = st.selectbox("País 1", country_stats[cols['country']].unique(), index=0)
    with col2:
        # Asegurarse de que el país 2 es diferente al país 1
        otros_paises = [p for p in country_stats[cols['country']].unique() if p != pais1]
        index_pais2 = 0 if len(otros_paises) > 0 else 0
        pais2 = st.selectbox("País 2", otros_paises, index=index_pais2)
    
    if pais1 and pais2:
        # Obtener datos para ambos países
        data_pais1 = country_stats[country_stats[cols['country']] == pais1].iloc[0]
        data_pais2 = country_stats[country_stats[cols['country']] == pais2].iloc[0]
        
        n1, n2 = data_pais1[cols['confirmed']], data_pais2[cols['confirmed']]
        d1, d2 = data_pais1[cols['deaths']], data_pais2[cols['deaths']]
        
        # Realizar test de proporciones z-test
        if n1 > 0 and n2 > 0:
            p1 = d1 / n1
            p2 = d2 / n2
            p_pool = (d1 + d2) / (n1 + n2)
            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z_score = (p1 - p2) / se_pool
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            st.write(f"**Resultado del test de hipótesis:**")
            st.write(f"- Proporción {pais1}: {p1*100:.2f}%")
            st.write(f"- Proporción {pais2}: {p2*100:.2f}%")
            st.write(f"- Diferencia: {(p1-p2)*100:.2f}%")
            st.write(f"- Z-score: {z_score:.4f}")
            st.write(f"- Valor p: {p_value:.4f}")
            
            if p_value < 0.05:
                st.success("La diferencia es estadísticamente significativa (p < 0.05)")
            else:
                st.warning("La diferencia no es estadísticamente significativa (p ≥ 0.05)")
        else:
            st.warning("Datos insuficientes para realizar el test de hipótesis")
    
    # 2.4. Detección de outliers usando Z-score
    st.subheader("Detección de Outliers (Z-score)")
    
    # Calcular Z-scores para las métricas
    for metric in [cols['confirmed'], cols['deaths']]:
        metric_values = country_stats[metric]
        z_scores = np.abs(stats.zscore(metric_values))
        outliers = country_stats[z_scores > 3]  # Z-score > 3 se considera outlier
        
        if not outliers.empty:
            st.write(f"**Outliers en {metric} (Z-score > 3):**")
            st.dataframe(outliers[[cols['country'], metric]])
        else:
            st.write(f"No se encontraron outliers en {metric}")
    
    # 2.5. Gráfico de control (3σ) de muertes diarias
    st.subheader("Gráfico de Control para Muertes Diarias")
    
    # Cargar series temporales para el gráfico de control
    try:
        deaths_ts = load_time_series("deaths")
        
        # Agregar por fecha
        daily_deaths = deaths_ts.groupby('Date')['Deaths'].sum().reset_index()
        
        # Calcular media y desviación estándar
        mean_deaths = daily_deaths['Deaths'].mean()
        std_deaths = daily_deaths['Deaths'].std()
        
        # Límites de control
        ucl = mean_deaths + 3 * std_deaths  # Límite superior de control
        lcl = max(0, mean_deaths - 3 * std_deaths)  # Límite inferior de control (no negativo)
        
        # Crear gráfico de control
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_deaths['Date'], y=daily_deaths['Deaths'], 
                                mode='lines', name='Muertes diarias'))
        fig.add_trace(go.Scatter(x=daily_deaths['Date'], y=[mean_deaths] * len(daily_deaths), 
                                mode='lines', name='Media', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=daily_deaths['Date'], y=[ucl] * len(daily_deaths), 
                                mode='lines', name='LSC (3σ)', line=dict(dash='dot', color='red')))
        fig.add_trace(go.Scatter(x=daily_deaths['Date'], y=[lcl] * len(daily_deaths), 
                                mode='lines', name='LIC (3σ)', line=dict(dash='dot', color='red')))
        
        # Resaltar puntos fuera de control
        outliers = daily_deaths[(daily_deaths['Deaths'] > ucl) | (daily_deaths['Deaths'] < lcl)]
        if not outliers.empty:
            fig.add_trace(go.Scatter(x=outliers['Date'], y=outliers['Deaths'], 
                                    mode='markers', name='Puntos fuera de control',
                                    marker=dict(color='red', size=8)))
        
        fig.update_layout(title='Gráfico de Control para Muertes Diarias (3σ)',
                         xaxis_title='Fecha',
                         yaxis_title='Número de Muertes')
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar puntos fuera de control
        if not outliers.empty:
            st.write("**Puntos fuera de control:**")
            st.dataframe(outliers)
        else:
            st.info("No se encontraron puntos fuera de control en el período analizado.")
            
    except Exception as e:
        st.error(f"No se pudieron cargar las series temporales: {str(e)}")

# ---------------------------------------------------------------
# Pestaña 3: Modelado Temporal
# ---------------------------------------------------------------
with tab3:
    st.header("Modelado Temporal y Pronósticos")
    
    # 3.1. Series de tiempo con suavizado de 7 días
    st.subheader("Series de Tiempo con Suavizado")
    
    # Cargar series temporales
    try:
        confirmed_ts = load_time_series("confirmed")
        deaths_ts = load_time_series("deaths")
        
        # Filtrar por país seleccionado
        if pais_seleccionado:
            confirmed_ts = confirmed_ts[confirmed_ts['Country/Region'].isin(pais_seleccionado)]
            deaths_ts = deaths_ts[deaths_ts['Country/Region'].isin(pais_seleccionado)]
        
        # Agregar por fecha
        daily_confirmed = confirmed_ts.groupby('Date')['Confirmed'].sum().reset_index()
        daily_deaths = deaths_ts.groupby('Date')['Deaths'].sum().reset_index()
        
        # Calcular media móvil de 7 días
        daily_confirmed['MA7'] = daily_confirmed['Confirmed'].rolling(window=7).mean()
        daily_deaths['MA7'] = daily_deaths['Deaths'].rolling(window=7).mean()
        
        # Crear gráfico de series temporales
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Casos Confirmados', 'Muertes'))
        
        # Casos confirmados
        fig.add_trace(go.Scatter(x=daily_confirmed['Date'], y=daily_confirmed['Confirmed'],
                                mode='lines', name='Confirmados diarios', opacity=0.3),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=daily_confirmed['Date'], y=daily_confirmed['MA7'],
                                mode='lines', name='Media móvil 7d'),
                     row=1, col=1)
        
        # Muertes
        fig.add_trace(go.Scatter(x=daily_deaths['Date'], y=daily_deaths['Deaths'],
                                mode='lines', name='Muertes diarias', opacity=0.3),
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=daily_deaths['Date'], y=daily_deaths['MA7'],
                                mode='lines', name='Media móvil 7d'),
                     row=2, col=1)
        
        fig.update_layout(height=600, title_text="Evolución Temporal con Suavizado")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error al cargar series temporales: {str(e)}")
    
    # 3.2. Modelo de pronóstico (SARIMA simplificado)
    st.subheader("Pronóstico a 14 Días")
    
    # Seleccionar variable para pronóstico
    forecast_var = st.selectbox("Variable para pronóstico", ["Confirmed", "Deaths"])
    
    # Preparar datos para el modelo
    if forecast_var == "Confirmed":
        ts_data = daily_confirmed.dropna().set_index('Date')['MA7']
    else:
        ts_data = daily_deaths.dropna().set_index('Date')['MA7']
    
    if len(ts_data) > 30:  # Necesitamos suficientes datos para el modelo
        # Dividir en train y test
        train_size = int(len(ts_data) * 0.8)
        train, test = ts_data.iloc[:train_size], ts_data.iloc[train_size:]
        
        # Entrenar modelo ARIMA (simplificado)
        try:
            model = ARIMA(train, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Pronóstico
            forecast_steps = 14
            forecast = model_fit.forecast(steps=forecast_steps)
            forecast_index = pd.date_range(start=ts_data.index[-1] + timedelta(days=1), periods=forecast_steps)
            
            # 3.4. Bandas de confianza (aproximación)
            # Para simplificar, usamos un intervalo basado en el error histórico
            last_mae = np.mean(np.abs(model_fit.resid.dropna()))
            upper_band = forecast + 1.96 * last_mae
            lower_band = forecast - 1.96 * last_mae
            lower_band = lower_band.apply(lambda x: max(0, x))  # No valores negativos
            
            # 3.3. Validación con backtesting (MAE/MAPE)
            # Predecir para el período de test
            predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
            mae = np.mean(np.abs(predictions - test))
            mape = np.mean(np.abs((predictions - test) / test)) * 100
            
            st.write(f"**Métricas de Validación:**")
            st.write(f"- MAE: {mae:.2f}")
            st.write(f"- MAPE: {mape:.2f}%")
            
            # Visualizar pronóstico
            fig = go.Figure()
            
            # Datos históricos
            fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values,
                                    mode='lines', name='Histórico'))
            
            # Período de test
            fig.add_trace(go.Scatter(x=test.index, y=test.values,
                                    mode='lines', name='Real (período test)'))
            fig.add_trace(go.Scatter(x=test.index, y=predictions,
                                    mode='lines', name='Predicción (período test)'))
            
            # Pronóstico futuro
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast,
                                    mode='lines', name='Pronóstico 14d'))
            fig.add_trace(go.Scatter(x=forecast_index, y=upper_band,
                                    mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_index, y=lower_band,
                                    mode='lines', line=dict(width=0),
                                    fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                                    name='Intervalo confianza'))
            
            fig.update_layout(title=f"Pronóstico de {forecast_var} para próximos 14 días",
                             yaxis_title=forecast_var)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en el modelo: {str(e)}")
    else:
        st.warning("No hay suficientes datos para generar un pronóstico confiable")

# ---------------------------------------------------------------
# Pestaña 4: Clustering y PCA
# ---------------------------------------------------------------
with tab4:
    st.header("Segmentación de Países mediante Clustering")
    
    # Preparar datos para clustering
    clustering_data = country_stats.copy()
    
    # Calcular tasa de crecimiento (aproximación)
    # Para simplificar, usaremos una tasa fija ya que no tenemos histórico en esta pestaña
    clustering_data['Crecimiento_7d'] = np.random.uniform(0, 5, len(clustering_data))  # Placeholder
    
    # Seleccionar variables para clustering
    cluster_vars = ['CFR', 'Tasa_100k', 'Crecimiento_7d']
    X = clustering_data[cluster_vars].dropna()
    
    if len(X) > 1:
        # Estandarizar variables
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 4.1. Clustering con K-means
        n_clusters = st.slider("Número de clusters", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Añadir clusters al dataframe
        clustering_data['Cluster'] = np.nan
        clustering_data.loc[X.index, 'Cluster'] = clusters
        
        # 4.2. Aplicar PCA para reducción a 2 componentes
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Crear dataframe para visualización
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = clusters
        pca_df['Country'] = clustering_data.loc[X.index, cols['country']].values
        pca_df['CFR'] = clustering_data.loc[X.index, 'CFR'].values
        
        # Visualizar clusters
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                         hover_data=['Country', 'CFR'],
                         title=f"PCA - Clusters de Países (K={n_clusters})")
        st.plotly_chart(fig, use_container_width=True)
        
        # 4.3. Interpretación de clusters
        st.subheader("Interpretación de Clusters")
        
        # Calcular estadísticas por cluster
        cluster_stats = clustering_data.dropna().groupby('Cluster')[cluster_vars].mean()
        st.dataframe(cluster_stats)
        
        # Descripción de cada cluster
        for cluster_id in range(n_clusters):
            cluster_countries = clustering_data[clustering_data['Cluster'] == cluster_id][cols['country']].tolist()
            st.write(f"**Cluster {cluster_id}:** {', '.join(cluster_countries[:5])}{'...' if len(cluster_countries) > 5 else ''}")
            
            # Caracterizar cluster según sus estadísticas
            cfr_mean = cluster_stats.loc[cluster_id, 'CFR']
            tasa_mean = cluster_stats.loc[cluster_id, 'Tasa_100k']
            crecimiento_mean = cluster_stats.loc[cluster_id, 'Crecimiento_7d']
            
            caracterizacion = f"CFR: {cfr_mean:.2f}%, Tasa por 100k: {tasa_mean:.2f}, Crecimiento 7d: {crecimiento_mean:.2f}%"
            st.caption(caracterizacion)
    else:
        st.warning("No hay suficientes datos para realizar clustering")

# ---------------------------------------------------------------
# Pestaña 5: Calidad de Datos
# ---------------------------------------------------------------
with tab5:
    st.header("Calidad y Consistencia de Datos")
    
    # Análisis de valores nulos
    st.subheader("Valores Nulos por Columna")
    null_analysis = df.isnull().sum().reset_index()
    null_analysis.columns = ['Columna', 'Valores Nulos']
    null_analysis['Porcentaje'] = (null_analysis['Valores Nulos'] / len(df)) * 100
    
    fig = px.bar(null_analysis, x='Columna', y='Porcentaje', 
                 title='Porcentaje de Valores Nulos por Columna')
    st.plotly_chart(fig, use_container_width=True)
    
    # Inconsistencias en datos
    st.subheader("Posibles Inconsistencias")
    
    # Verificar que las muertes no superen los casos confirmados
    inconsistent_deaths = df[df[cols['deaths']] > df[cols['confirmed']]]
    if not inconsistent_deaths.empty:
        st.warning(f"Se encontraron {len(inconsistent_deaths)} registros donde las muertes superan los casos confirmados")
        st.dataframe(inconsistent_deaths[[cols['country'], cols['province'], cols['confirmed'], cols['deaths']]])
    else:
        st.success("No se encontraron inconsistencias en la relación casos confirmados/muertes")
    
    # Verificar valores negativos
    negative_values = {}
    for col in [cols['confirmed'], cols['deaths']]:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            negative_values[col] = negative_count
    
    if negative_values:
        st.warning("Se encontraron valores negativos en las siguientes columnas:")
        st.write(negative_values)
    else:
        st.success("No se encontraron valores negativos en las métricas principales")

# ---------------------------------------------------------------
# Pestaña 6: Propuesta de Startup
# ---------------------------------------------------------------
with tab6:
    st.header("🚀 Propuesta de Startup: HealthInsight AI")
    
    st.markdown("""
    ### 🎯 Identificación del Problema
    La gestión sanitaria durante pandemias enfrenta múltiples desafíos:
    - **Detección tardía de brotes**: Los sistemas tradicionales identifican patrones demasiado tarde
    - **Asignación ineficiente de recursos**: Limitaciones en la distribución de insumos médicos
    - **Falta de estandarización**: Diferentes criterios para semáforos epidemiológicos
    - **Saturación de sistemas de salud**: Imposibilidad de predecir picos de demanda
    """)
    
    st.markdown("""
    ### 💡 Solución Digital: HealthInsight AI
    Plataforma integral de inteligencia epidemiológica que combina:
    - **Vigilancia en tiempo real**: Monitoreo automático de múltiples fuentes de datos
    - **Predictores avanzados**: Modelos de machine learning para anticipar brotes
    - **Sistema de alertas tempranas**: Notificaciones proactivas a autoridades sanitarias
    - **Dashboard unificado**: Visualización intuitiva para la toma de decisiones
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 👥 Clientes y Aliados Estratégicos
        - **MINSA**: Implementación a nivel nacional
        - **EsSalud**: Optimización de recursos hospitalarios
        - **Municipalidades**: Gestión local de semáforos epidemiológicos
        - **Clínicas Privadas**: Planificación de capacidad y recursos
        - **Organismos Internacionales**: OPS/OMS para vigilancia regional
        """)
    
    with col2:
        st.markdown("""
        ### 📊 MVP (Producto Mínimo Viable)
        - **Módulo de visualización**: Dashboard con los indicadores clave
        - **Sistema de alertas**: Notificaciones por correo/API
        - **Reportes automáticos**: Generación diaria de reportes epidemiológicos
        - **API básica**: Acceso programático a los datos procesados
        """)
    
    st.markdown("""
    ### 🌟 Métrica Estrella (North Star)
    **Tiempo de detección temprana**: Reducción del tiempo entre el inicio de un brote y su identificación por las autoridades sanitarias.
    
    #### 📈 KPIs de Impacto
    - Reducción del 40% en tiempo de detección de brotes
    - Disminución del 25% en mortalidad por respuesta temprana
    - Optimización del 30% en asignación de recursos
    - Aumento del 50% en capacidad predictiva
    """)
    
    st.markdown("""
    ### 💰 Modelo de Negocio
    - **Suscripción anual**: Por institución según tamaño y necesidades
    - **API premium**: Pago por consumo de datos y predicciones
    - **Licenciamiento**: Implementación white-label para gobiernos
    - **Consultoría**: Servicios personalizados de análisis y implementación
    
    **Precios estimados**:
    - Municipalidades: $5,000-$20,000/año
    - Hospitales: $10,000-$50,000/año  
    - Gobiernos regionales: $50,000-$200,000/año
    """)
    
    st.markdown("""
    ### 🗺️ Roadmap en Fases
    """)
    
    roadmap_data = {
        'Fase': ['Fase 1 (0-6 meses)', 'Fase 2 (6-12 meses)', 'Fase 3 (12-18 meses)', 'Fase 4 (18-24 meses)'],
        'Objetivo': [
            'Vigilancia básica y dashboard',
            'Predicción y alertas tempranas', 
            'Interoperabilidad con sistemas existentes',
            'Expansión internacional y nuevas enfermedades'
        ],
        'Características': [
            'Visualización de datos, reportes automáticos',
            'Modelos predictivos, sistema de alertas',
            'APIs, integración con HIS, mobile apps',
            'Multi-enfermedades, multi-país, IA avanzada'
        ]
    }
    
    roadmap_df = pd.DataFrame(roadmap_data)
    st.table(roadmap_df)
    
    st.markdown("""
    ### 📋 Próximos Pasos
    1. **Validación con MINSA**: Presentación de MVP y ajuste de requerimientos
    2. **Piloto en Lima Metropolitana**: Implementación inicial en 3 distritos
    3. **Ronda de financiamiento**: Búsqueda de $500k para escalamiento
    4. **Equipo técnico**: Ampliación a 10 personas (datos, desarrollo, epidemiología)
    """)

# ---------------------------------------------------------------
# Exportación de datos
# ---------------------------------------------------------------
st.sidebar.header("Exportación de Datos")
if st.sidebar.button("Exportar Datos a CSV"):
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label="Descargar CSV",
        data=csv,
        file_name=f"covid_data_{fecha_str}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------------
# Narrativa automática
# ---------------------------------------------------------------
st.sidebar.header("Narrativa Automática")
if st.sidebar.button("Generar Resumen"):
    # KPIs para la narrativa
    total_countries = df[cols['country']].nunique()
    max_deaths_country = country_stats.loc[country_stats[cols['deaths']].idxmax()][cols['country']]
    max_deaths_value = country_stats[cols['deaths']].max()
    
    # Encontrar país con mayor CFR (excluyendo divisiones por cero)
    country_stats_valid = country_stats[country_stats['CFR'].notna()]
    if not country_stats_valid.empty:
        max_cfr_country = country_stats_valid.loc[country_stats_valid['CFR'].idxmax()][cols['country']]
        max_cfr_value = country_stats_valid['CFR'].max()
    else:
        max_cfr_country = "N/A"
        max_cfr_value = 0
    
    max_null_col = null_analysis.loc[null_analysis['Porcentaje'].idxmax(), 'Columna']
    max_null_pct = null_analysis['Porcentaje'].max()
    
    narrative = f"""
    ## Resumen de Análisis COVID-19 al {fecha_str}
    
    El análisis de los datos de COVID-19 revela que, hasta la fecha seleccionada, se han reportado 
    **{total_confirmados:,} casos confirmados** y **{total_fallecidos:,} fallecidos** a nivel global, 
    con una tasa de mortalidad general del **{tasa_mortalidad:.2f}%**.
    
    Los datos cubren **{total_countries} países**, siendo **{max_deaths_country}** el país con mayor número 
    de fallecidos reportados ({max_deaths_value:,}). En términos de tasa de mortalidad, **{max_cfr_country}** 
    presenta la CFR más alta ({max_cfr_value:.2f}%).
    
    El análisis de calidad de datos muestra que el {max_null_pct:.2f}% de valores nulos 
    se encuentra en la columna '{max_null_col}'.
    """
    
    st.sidebar.markdown(narrative)

# ---------------------------------------------------------------
# Funcionalidades originales (mantenidas por compatibilidad)
# ---------------------------------------------------------------
