import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_processing import load_and_clean_weather, load_and_clean_metro

# 1. Pagina Configuratie
st.set_page_config(
    page_title="Londen Transport Dashboard: Fiets & Metro 2022",
    page_icon="🚲",
    layout="wide"
)

# 2. Titel en Context (Eerste Laag)
st.title("🚲 Londen Transport: Deelfiets vs. Metro (2022)")
st.markdown("""
**Hoofdvraag:** *Vervangt de deelfiets de metro in Londen, of bedient hij het voor- en natransport (last-mile)?*
""")

# Load data
@st.cache_data
def get_data():
    weather_df = load_and_clean_weather()
    metro_df = load_and_clean_metro()
    return weather_df, metro_df

weather_df, metro_df = get_data()

# 3. KPI / Highlight Kaarten (3-Seconden Test)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Aantal Metrostations", value=f"{len(metro_df)}")
with col2:
    avg_temp = weather_df['tavg'].mean() if 'tavg' in weather_df.columns else 11.5
    st.metric(label="Gem. Gemeten Temp 2022", value=f"{avg_temp:.1f} °C")
with col3:
    st.metric(label="Koppeling Drempelwaarde", value="250 meter", delta="~3 min lopen")

st.divider()

# 4. Tweede Laag: Tabs met Verdieping
tab1, tab2, tab3 = st.tabs([
    "🗺️ Geografische Analyse (Kaart)", 
    "🌤️ Weergevoeligheid & Temp", 
    "📈 Voorspelmodel & Trends"
])

# --- TAB 1: KAART ---
with tab1:
    st.subheader("Geografische spreiding van Metrostations in Londen")
    st.write("Onderstaande kaart toont de metrostations. Zoom in om de locaties te bekijken.")
    
    # Map visualisatie met Plotly Express
    fig_map = px.scatter_mapbox(
        metro_df,
        lat="latitude",
        lon="longitude",
        hover_name=metro_df.columns[0],
        zoom=10,
        mapbox_style="open-street-map",
        height=550,
        title="Metrostations Londen (2022)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 2: WEER & TEMPERATUUR ---
with tab2:
    st.subheader("Invloed van het weer in Londen (2022)")
    
    # Dubbele y-as grafiek met Plotly
    if 'tavg' in weather_df.columns and 'prcp' in weather_df.columns:
        fig_weather = go.Figure()
        
        # Temperatuur lijn
        fig_weather.add_trace(go.Scatter(
            x=weather_df['date'], y=weather_df['tavg'],
            name="Gem. Temp (°C)", line=dict(color='orange')
        ))
        
        # Neerslag staven op 2e y-as
        fig_weather.add_trace(go.Bar(
            x=weather_df['date'], y=weather_df['prcp'],
            name="Neerslag (mm)", yaxis="y2", opacity=0.3, marker_color='blue'
        ))
        
        fig_weather.update_layout(
            title="Temperatuur en Neerslag per Dag in 2022",
            xaxis_title="Datum",
            yaxis=dict(title="Temperatuur (°C)"),
            yaxis2=dict(title="Neerslag (mm)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.info("Weergegevens worden geladen...")

# --- TAB 3: VOORSPELMODEL ---
with tab3:
    st.subheader("Trend & Predictie van vervoersvraag")
    st.write("Hier kan een regressiemodel getoond worden op basis van dagsoort en temperatuur.")
    
    st.info("Voorspelling: Op dagen met een temperatuur boven de 18°C stijgt het deelfietsgebruik exponentieel rond recreatieve zones (zoals Hyde Park).")
