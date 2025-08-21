import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, proportions_ztest
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports"

@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
    yyyy, mm, dd = yyyy_mm_dd.split("-")
    url = f"{GITHUB_BASE}/{mm}-{dd}-{yyyy}.csv"
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
    }
    return df, url, cols

@st.cache_data
def get_population_data():
    """Datos de población por país (en millones)"""
    return {
        'US': 332.4, 'China': 1412.0, 'India': 1380.0, 'Indonesia': 275.5,
        'Pakistan': 225.2, 'Brazil': 215.3, 'Nigeria': 218.5, 'Bangladesh': 166.3,
        'Russia': 144.1, 'Mexico': 130.3, 'Japan': 125.8, 'Philippines': 111.0,
        'Ethiopia': 117.9, 'Vietnam': 98.2, 'Egypt': 104.3, 'Germany': 83.2,
        'Iran': 85.0, 'Turkey': 84.8, 'Thailand': 70.0, 'United Kingdom': 67.5,
        'France': 67.8, 'Italy': 59.1, 'Tanzania': 61.5, 'South Africa': 60.4,
        'Myanmar': 54.8, 'Kenya': 54.0, 'South Korea': 51.7, 'Colombia': 51.0,
        'Spain': 47.4, 'Uganda': 47.1, 'Argentina': 45.6, 'Algeria': 44.9,
        'Sudan': 44.9, 'Ukraine': 43.8, 'Iraq': 41.2, 'Afghanistan': 39.8,
        'Poland': 37.8, 'Canada': 38.2, 'Morocco': 37.5, 'Saudi Arabia': 35.0,
        'Uzbekistan': 34.9, 'Peru': 33.0, 'Angola': 33.9, 'Malaysia': 32.7,
        'Chile': 19.5, 'Romania': 19.1, 'Netherlands': 17.4, 'Ecuador': 17.9,
        'Belgium': 11.6, 'Cuba': 11.3, 'Greece': 10.7, 'Portugal': 10.3,
        'Czech Republic': 10.7, 'Hungary': 9.7, 'Sweden': 10.4, 'Austria': 9.0,
        'Switzerland': 8.7, 'Israel': 9.4, 'Denmark': 5.8, 'Finland': 5.5,
        'Slovakia': 5.5, 'Norway': 5.4, 'Ireland': 5.0, 'New Zealand': 5.1,
        'Croatia': 3.9, 'Uruguay': 3.5, 'Lithuania': 2.8, 'Slovenia': 2.1,
        'Latvia': 1.9, 'Estonia': 1.3, 'Cyprus': 1.2, 'Luxembourg': 0.6,
        'Malta': 0.5, 'Iceland': 0.4
    }

def intervalo_confianza_cfr(muertes, casos, confianza=0.95):
    """Calcula intervalo de confianza para CFR usando distribución binomial"""
    if casos == 0:
        return np.nan, np.nan
    
    p = muertes / casos
    z = norm.ppf((1 + confianza) / 2)
    error_std = np.sqrt(p * (1 - p) / casos)
    
    ic_inferior = max(0, (p - z * error_std) * 100)
    ic_superior = min(100, (p + z * error_std) * 100)
    
    return ic_inferior, ic_superior

def detectar_outliers_zscore(data, columna, umbral=3):
    """Detectar outliers usando Z-score"""
    valores = data[columna].dropna()
    if len(valores) == 0:
        return pd.DataFrame()
    z_scores = np.abs(stats.zscore(valores))
    outlier_mask = z_scores > umbral
    return data[data[columna].isin(valores[outlier_mask])]

def detectar_outliers_iqr(data, columna):
    """Detectar outliers usando IQR"""
    Q1 = data[columna].quantile(0.25)
    Q3 = data[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return data[(data[columna] < limite_inferior) | (data[columna] > limite_superior)]

st.sidebar.title("Opciones")
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")

try:
    df, source_url, cols = load_daily_report(fecha_str)
    st.sidebar.success(f"✅ Datos cargados: {len(df)} registros")
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")
    st.stop()

st.sidebar.caption(f"Fuente: {source_url}")

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# ———————————————————————————————————————————————
# a) Mostrar todas las filas del dataset, luego volver al estado inicial
# ———————————————————————————————————————————————
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# ———————————————————————————————————————————————
# b) Mostrar todas las columnas del dataset
# ———————————————————————————————————————————————
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))

st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# ———————————————————————————————————————————————
# c) Línea del total de fallecidos (>2500) vs Confirmed/Recovered/Active por país
# ———————————————————————————————————————————————
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

# ———————————————————————————————————————————————
# d) Barras de fallecidos de estados de Estados Unidos
# ———————————————————————————————————————————————
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = cols["country"]
prov_col = cols["province"]

dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# ———————————————————————————————————————————————
# e) Gráfica de sectores (simulada con barra si no hay pie nativo)
# ———————————————————————————————————————————————
st.header("e) Gráfica de sectores (simulada)")
lista_paises = ["Colombia", "Chile", "Peru", "Argentina", "Mexico"]
sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), default=lista_paises)
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    # Como Streamlit no tiene pie nativo, mostramos distribución normalizada como barra
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# ———————————————————————————————————————————————
# f) Histograma del total de fallecidos por país (simulado con bar_chart)
# ———————————————————————————————————————————————
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# ———————————————————————————————————————————————
# g) Boxplot de Confirmed, Deaths, Recovered, Active (simulado con box_chart)
# ———————————————————————————————————————————————
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
# Streamlit no tiene boxplot nativo, así que mostramos estadísticas resumen en tabla
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)

# ==================== ESTADÍSTICA DESCRIPTIVA Y AVANZADA ====================
st.markdown("---")
st.title("📊 2. Estadística Descriptiva y Avanzada")

# Preparar datos agregados por país
by_pais = df.groupby(country_col)[metrics].sum().reset_index()
by_pais = by_pais.rename(columns={country_col: "Country_Region"})

# ==================== 2.1 MÉTRICAS CLAVE ====================
st.header("2.1 📈 Métricas Clave por País")

# Calcular CFR
by_pais['CFR'] = np.where(by_pais[C] > 0, (by_pais[D] / by_pais[C]) * 100, 0)

# Agregar datos de población y calcular tasas por 100k
poblaciones = get_population_data()
by_pais['Poblacion_Mill'] = by_pais['Country_Region'].map(poblaciones)
by_pais['Confirmados_100k'] = np.where(by_pais['Poblacion_Mill'].notna(),
                                      (by_pais[C] / (by_pais['Poblacion_Mill'] * 10000)), np.nan)
by_pais['Muertes_100k'] = np.where(by_pais['Poblacion_Mill'].notna(),
                                  (by_pais[D] / (by_pais['Poblacion_Mill'] * 10000)), np.nan)

# Métricas globales
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Confirmados", f"{by_pais[C].sum():,}")
with col2:
    st.metric("Total Fallecidos", f"{by_pais[D].sum():,}")
with col3:
    cfr_global = (by_pais[D].sum() / by_pais[C].sum()) * 100 if by_pais[C].sum() > 0 else 0
    st.metric("CFR Global", f"{cfr_global:.2f}%")
with col4:
    st.metric("Países analizados", f"{len(by_pais)}")

# Top países
st.subheader("Top 15 países con métricas clave")
top_paises = by_pais.nlargest(15, C)
cols_mostrar = ['Country_Region', C, D, 'CFR', 'Confirmados_100k', 'Muertes_100k']
st.dataframe(top_paises[cols_mostrar].round(2))

# Estadísticas del CFR
paises_significativos = by_pais[by_pais[C] >= 1000]
if len(paises_significativos) > 0:
    st.subheader("📊 Estadísticas del CFR (países con ≥1000 casos)")
    cfr_stats = paises_significativos['CFR']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CFR Promedio", f"{cfr_stats.mean():.2f}%")
    with col2:
        st.metric("CFR Mediana", f"{cfr_stats.median():.2f}%")
    with col3:
        st.metric("Desv. Estándar", f"{cfr_stats.std():.2f}%")
    with col4:
        st.metric("Rango", f"{cfr_stats.min():.2f}% - {cfr_stats.max():.2f}%")

# ==================== 2.2 INTERVALOS DE CONFIANZA ====================
st.header("2.2 📊 Intervalos de Confianza para CFR")

confianza = st.select_slider("Nivel de confianza", options=[90, 95, 99], value=95)
n_paises_ic = st.slider("Número de países a analizar", 5, 20, 10)

top_paises_ic = by_pais.nlargest(n_paises_ic, C).copy()
ic_resultados = []

for _, pais in top_paises_ic.iterrows():
    ic_inf, ic_sup = intervalo_confianza_cfr(pais[D], pais[C], confianza/100)
    ic_resultados.append({
        'País': pais['Country_Region'],
        'CFR': pais['CFR'],
        'IC_Inferior': ic_inf,
        'IC_Superior': ic_sup,
        'Casos': pais[C],
        'Muertes': pais[D]
    })

ic_df = pd.DataFrame(ic_resultados)
st.subheader(f"Intervalos de Confianza del CFR ({confianza}%)")
st.dataframe(ic_df.round(3))

# Visualización de IC con Plotly
fig_ic = go.Figure()

for i, row in ic_df.iterrows():
    fig_ic.add_trace(go.Scatter(
        x=[row['País']],
        y=[row['CFR']],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[row['IC_Superior'] - row['CFR']],
            arrayminus=[row['CFR'] - row['IC_Inferior']]
        ),
        mode='markers',
        marker=dict(size=10),
        name=row['País'],
        showlegend=False
    ))

fig_ic.update_layout(
    title=f"Intervalos de Confianza del CFR ({confianza}%)",
    xaxis_title="País",
    yaxis_title="CFR (%)",
    height=500
)
st.plotly_chart(fig_ic, use_container_width=True)

# ==================== 2.3 TEST DE HIPÓTESIS ====================
st.header("2.3 🔬 Test de Hipótesis de Proporciones")

paises_disponibles = by_pais[by_pais[C] >= 1000]['Country_Region'].tolist()

col1, col2 = st.columns(2)
with col1:
    pais1 = st.selectbox("Seleccionar país 1", paises_disponibles, index=0)
with col2:
    pais2 = st.selectbox("Seleccionar país 2", paises_disponibles, 
                        index=1 if len(paises_disponibles) > 1 else 0)

if pais1 != pais2:
    datos_pais1 = by_pais[by_pais['Country_Region'] == pais1].iloc[0]
    datos_pais2 = by_pais[by_pais['Country_Region'] == pais2].iloc[0]

    # Test de proporciones
    muertes = np.array([datos_pais1[D], datos_pais2[D]])
    casos = np.array([datos_pais1[C], datos_pais2[C]])

    try:
        z_stat, p_valor = proportions_ztest(muertes, casos)
        
        st.subheader("Resultados del Test Z de dos proporciones")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{pais1}:**")
            st.write(f"CFR = {datos_pais1['CFR']:.3f}%")
            st.write(f"Casos: {datos_pais1[C]:,}")
            st.write(f"Muertes: {datos_pais1[D]:,}")
        
        with col2:
            st.write(f"**{pais2}:**")
            st.write(f"CFR = {datos_pais2['CFR']:.3f}%")
            st.write(f"Casos: {datos_pais2[C]:,}")
            st.write(f"Muertes: {datos_pais2[D]:,}")
        
        st.write("**Hipótesis:**")
        st.write(f"H₀: CFR_{pais1} = CFR_{pais2}")
        st.write(f"H₁: CFR_{pais1} ≠ CFR_{pais2}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estadístico Z", f"{z_stat:.4f}")
        with col2:
            st.metric("P-valor", f"{p_valor:.6f}")
        with col3:
            resultado = "Sí" if p_valor < 0.05 else "No"
            st.metric("¿Diferencia significativa? (α=0.05)", resultado)
        
        if p_valor < 0.05:
            st.success("✅ **Conclusión:** Rechazamos H₀. Hay diferencia significativa entre los CFR.")
        else:
            st.info("ℹ️ **Conclusión:** No rechazamos H₀. No hay diferencia significativa entre los CFR.")
    
    except Exception as e:
        st.error(f"Error en el test: {e}")

# ==================== 2.4 DETECCIÓN DE OUTLIERS ====================
st.header("2.4 🎯 Detección de Outliers")

# Filtrar países con casos significativos
paises_filtrados = by_pais[by_pais[C] >= 100].copy()

st.subheader("Outliers en CFR")
metodo_outlier = st.radio("Seleccionar método", ["Z-score", "IQR"], horizontal=True)

if metodo_outlier == "Z-score":
    umbral_z = st.slider("Umbral Z-score", 2.0, 4.0, 3.0, 0.1)
    outliers_cfr = detectar_outliers_zscore(paises_filtrados, 'CFR', umbral_z)
else:
    outliers_cfr = detectar_outliers_iqr(paises_filtrados, 'CFR')

if len(outliers_cfr) > 0:
    st.write(f"**Outliers detectados ({metodo_outlier}):** {len(outliers_cfr)} países")
    st.dataframe(outliers_cfr[['Country_Region', 'CFR', C, D]].round(3))
else:
    st.info(f"No se encontraron outliers con el método {metodo_outlier}")

# Visualización de outliers
fig_outliers = px.scatter(paises_filtrados, x=C, y='CFR', hover_name='Country_Region',
                         title="Scatter Plot: Casos Confirmados vs CFR",
                         log_x=True)
st.plotly_chart(fig_outliers, use_container_width=True)

# Boxplot del CFR
fig_box = px.box(paises_filtrados, y='CFR', title="Boxplot del CFR por país")
st.plotly_chart(fig_box, use_container_width=True)

# ==================== 2.5 GRÁFICO DE CONTROL ====================
st.header("2.5 📈 Gráfico de Control (3σ)")

st.info("📝 **Nota:** Como no disponemos de series temporales, usaremos los datos por región/provincia como simulación de observaciones secuenciales.")

# Crear datos para gráfico de control usando muertes por región
muertes_regionales = df[df[D] > 0][D].sort_values().reset_index(drop=True)

if len(muertes_regionales) > 10:  # Necesitamos suficientes datos
    # Límites de control
    media_muertes = muertes_regionales.mean()
    sigma_muertes = muertes_regionales.std()
    
    limite_superior = media_muertes + 3 * sigma_muertes
    limite_inferior = max(0, media_muertes - 3 * sigma_muertes)
    limite_alerta_sup = media_muertes + 2 * sigma_muertes
    limite_alerta_inf = max(0, media_muertes - 2 * sigma_muertes)
    
    # Mostrar estadísticas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Media", f"{media_muertes:.0f}")
        st.metric("Desv. Estándar", f"{sigma_muertes:.0f}")
    with col2:
        st.metric("Límite Control Superior", f"{limite_superior:.0f}")
        st.metric("Límite Control Inferior", f"{limite_inferior:.0f}")
    with col3:
        st.metric("Límite Alerta Superior", f"{limite_alerta_sup:.0f}")
        st.metric("Límite Alerta Inferior", f"{limite_alerta_inf:.0f}")
    
    # Tomar muestra para visualización
    n_muestra = st.slider("Número de observaciones a mostrar", 50, min(500, len(muertes_regionales)), 100)
    muestra = muertes_regionales.head(n_muestra)
    
    # Crear gráfico de control
    fig_control = go.Figure()
    
    # Datos observados
    fig_control.add_trace(go.Scatter(
        x=list(range(len(muestra))),
        y=muestra,
        mode='lines+markers',
        name='Observaciones',
        line=dict(color='blue')
    ))
    
    # Líneas de control
    fig_control.add_hline(y=media_muertes, line_dash="solid", line_color="green", 
                         annotation_text=f"Media ({media_muertes:.0f})")
    fig_control.add_hline(y=limite_superior, line_dash="dash", line_color="red",
                         annotation_text=f"LCS (3σ): {limite_superior:.0f}")
    fig_control.add_hline(y=limite_inferior, line_dash="dash", line_color="red",
                         annotation_text=f"LCI (3σ): {limite_inferior:.0f}")
    fig_control.add_hline(y=limite_alerta_sup, line_dash="dot", line_color="orange",
                         annotation_text=f"Alerta Sup (2σ): {limite_alerta_sup:.0f}")
    fig_control.add_hline(y=limite_alerta_inf, line_dash="dot", line_color="orange",
                         annotation_text=f"Alerta Inf (2σ): {limite_alerta_inf:.0f}")
    
    # Destacar puntos fuera de control
    puntos_fuera_control = muestra[(muestra > limite_superior) | (muestra < limite_inferior)]
    if len(puntos_fuera_control) > 0:
        indices_fuera = muestra.index[muestra.isin(puntos_fuera_control)]
        fig_control.add_trace(go.Scatter(
            x=indices_fuera,
            y=puntos_fuera_control,
            mode='markers',
            name='Fuera de control',
            marker=dict(color='red', size=12, symbol='x')
        ))
    
    fig_control.update_layout(
        title="Gráfico de Control (3σ) - Muertes por COVID-19",
        xaxis_title="Observación",
        yaxis_title="Número de muertes",
        height=500
    )
    
    st.plotly_chart(fig_control, use_container_width=True)
    
    # Análisis de puntos fuera de control
    st.subheader("Análisis de Control")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Puntos fuera de control", len(puntos_fuera_control))
        st.metric("% Fuera de control", f"{len(puntos_fuera_control)/len(muestra)*100:.1f}%")
    with col2:
        if len(puntos_fuera_control) > 0:
            st.write("**Valores fuera de control:**")
            st.write(puntos_fuera_control.values.tolist())
        else:
            st.success("✅ Todos los puntos están bajo control")

else:
    st.warning("Datos insuficientes para crear gráfico de control")

# ==================== RESUMEN FINAL ====================
st.markdown("---")
st.header("📋 Resumen del Análisis Estadístico")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("CFR Promedio Global", f"{cfr_global:.2f}%")
    st.metric("Países con IC calculado", len(ic_df))
with col2:
    if 'p_valor' in locals():
        diferencia = "Sí" if p_valor < 0.05 else "No"
        st.metric(f"Diferencia significativa ({pais1} vs {pais2})", diferencia)
    st.metric("Outliers detectados", len(outliers_cfr) if len(outliers_cfr) > 0 else 0)
with col3:
    if 'puntos_fuera_control' in locals():
        st.metric("Puntos fuera de control (3σ)", len(puntos_fuera_control))
    st.metric("Total países analizados", len(by_pais))

st.success("✅ **Análisis estadístico completado exitosamente!**")
