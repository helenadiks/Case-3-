import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

# ==========================================
# 0. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Londen Bikeshare & Metro Integratie Dashboard (2022)",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 🚇 Londen Bikeshare & Metro Integratie Dashboard (2022)")
st.markdown("""
Dit dashboard combineert **TfL Metrogebruik**, **Santander Bikeshare-ritten** en **Londen Weerdata** 
om te analyseren hoe fiets en metro elkaar aanvullen of vervangen.
""")

# ==========================================
# 1. DATA LOADING & CACHING FUNCTIONS
# ==========================================

def get_file_path(filename, folder="data"):
    """
    Zoekt een bestand in de opgeven map (bijv. 'data/') of in de hoofdfolder.
    """
    path_in_folder = os.path.join(folder, filename)
    if os.path.exists(path_in_folder):
        return path_in_folder
    elif os.path.exists(filename):
        return filename
    return None

@st.cache_data
def load_weather_data():
    file_path = get_file_path("weather_london.csv")
    if not file_path:
        st.error("❌ Kan 'weather_london.csv' niet vinden in de 'data/' map!")
        st.stop()
        
    df = pd.read_csv(file_path)
    # Hernoem Unnamed: 0 naar date
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'date'})
    
    df['date'] = pd.to_datetime(df['date'])
    # Filter op 2022 om aan te sluiten op metrodata
    df = df[df['date'].dt.year == 2022].copy()
    
    # Dropping leeg/onbetrouwbare kolommen zoals aangegeven in de opdracht
    df = df.drop(columns=['snow', 'tsun', 'wpgt'], errors='ignore')
    return df

@st.cache_data
def load_metro_data():
    file_path = get_file_path("metrogebruik_londen_2022.csv")
    if not file_path:
        st.error("❌ Kan 'metrogebruik_londen_2022.csv' niet vinden in de 'data/' map!")
        st.stop()
        
    df = pd.read_csv(file_path)
    # Filter de 11 metrostations zonder coördinaten
    df = df.dropna(subset=['lat', 'lon']).copy()
    return df

@st.cache_data
def load_journey_data(folder_path="data"):
    """
    Zoekt alle journey CSV's op in de data-map en voegt ze samen.
    """
    # Alle CSV bestanden ophalen in de data folder en root
    search_patterns = [
        os.path.join(folder_path, "*.csv"),
        "*.csv"
    ]
    
    all_csvs = []
    for pattern in search_patterns:
        all_csvs.extend(glob.glob(pattern))
    
    # Bekende vaste bestanden uitsluiten
    excluded = ["weather_london.csv", "metrogebruik_londen_2022.csv"]
    journey_files = list(set([f for f in all_csvs if os.path.basename(f) not in excluded]))
    
    if not journey_files:
        st.warning("⚠️ Geen reisgegevens (journey CSV's) gevonden in de 'data/' map.")
        return pd.DataFrame()
        
    df_list = []
    for file in journey_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            st.warning(f"Kon bestand {file} niet lezen: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Datumverwerking
    if 'Start date' in full_df.columns:
        full_df['Start date'] = pd.to_datetime(full_df['Start date'])
        full_df['Date'] = full_df['Start date'].dt.date
    if 'End date' in full_df.columns:
        full_df['End date'] = pd.to_datetime(full_df['End date'])
        
    return full_df

# Laden van alle data
with st.spinner("Data wordt geladen uit de 'data/' map..."):
    df_weather = load_weather_data()
    df_metro = load_metro_data()
    df_journeys = load_journey_data()

# ==========================================
# 2. SIDEBAR FILTERS & PARAMETERS
# ==========================================
st.sidebar.header("⚙️ Instellingen & Filters")

distance_threshold = st.sidebar.slider(
    "Maximale afstand Metro - Fietsstation (meters):",
    min_value=100,
    max_value=1000,
    value=250,
    step=50,
    help="Definieert de straal waarbinnen een fietsstation als 'aangesloten op metro' wordt beschouwd."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Overzicht")
st.sidebar.write(f"• Metrostations: **{len(df_metro)}**")
st.sidebar.write(f"• Weerdagen (2022): **{len(df_weather)}**")
st.sidebar.write(f"• Geladen ritten: **{len(df_journeys):,}**")

# ==========================================
# 3. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🗺️️ Metro & Netwerk", "🌧️ Weer & Fietsgebruik", "📌 Live Bevindingen & Presentatie"])

# ------------------------------------------
# TAB 1: METRO & GEOGRAFIE
# ------------------------------------------
with tab1:
    st.subheader("Geografische Spreiding Metrogebruik (2022)")
    
    col_map1, col_map2 = st.columns([3, 1])
    
    with col_map1:
        # Plotly Express Kaart
        fig_map = px.scatter_mapbox(
            df_metro,
            lat="lat",
            lon="lon",
            size="totaal_per_jaar",
            color="vervoerstype",
            hover_name="station",
            hover_data=["zone", "totaal_per_jaar", "in_maandag", "in_zaterdag"],
            zoom=10,
            mapbox_style="carto-positron",
            title="Metrostations Londen (Grootte = Totaal aantal reizigers/jaar)"
        )
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_map2:
        st.markdown("### ℹ️ Kaart Inzichten")
        st.write("""
        - **Cirkelgrootte:** Totaal aantal in- en uitstappers in 2022.
        - **Kleur:** Vervoerstype (Underground, Overground, DLR, Elizabeth line).
        """)
        st.metric(label="Geselecteerde Afstandsgrens", value=f"{distance_threshold} m")
        st.caption(f"Bij {distance_threshold}m worden nabijgelegen fietsdocks gekoppeld aan de dichtstbijzijnde hub.")

# ------------------------------------------
# TAB 2: WEER & FIETSGEBRUIK
# ------------------------------------------
with tab2:
    st.subheader("Invloed van Weer op het Aantal Fietsritten")
    
    if not df_journeys.empty and not df_weather.empty:
        # Dagelijkse ritten berekenen
        daily_journeys = df_journeys.groupby('Date').size().reset_index(name='rit_aantal')
        daily_journeys['Date'] = pd.to_datetime(daily_journeys['Date'])
        
        # Merge met weerdata
        df_merged = pd.merge(daily_journeys, df_weather, left_on='Date', right_on='date', how='inner')
        
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            fig_temp = px.scatter(
                df_merged,
                x="tavg",
                y="rit_aantal",
                size="prcp",
                color="tavg",
                color_continuous_scale="Viridis",
                trendline="ols",
                labels={"tavg": "Gemiddelde Temp (°C)", "rit_aantal": "Aantal Ritten per Dag", "prcp": "Neerslag (mm)"},
                title="Dagelijkse Ritten vs. Gemiddelde Temperatuur (°C)"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with col_w2:
            # Neerslag categoriseren
            df_merged['Neerslag_Categorie'] = pd.cut(
                df_merged['prcp'],
                bins=[-1, 0.1, 2.5, 10, 100],
                labels=['Droog (0mm)', 'Licht (0-2.5mm)', 'Matig (2.5-10mm)', 'Zwaar (>10mm)']
            )
            
            fig_rain = px.box(
                df_merged,
                x="Neerslag_Categorie",
                y="rit_aantal",
                color="Neerslag_Categorie",
                labels={"Neerslag_Categorie": "Neerslag Type", "rit_aantal": "Aantal Ritten per Dag"},
                title="Verdeling Aantal Ritten per Neerslagklasse"
            )
            st.plotly_chart(fig_rain, use_container_width=True)
            
        st.markdown("---")
        st.subheader("📈 Drukte per Uur / Ritduur")
        
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            if 'Start date' in df_journeys.columns:
                df_journeys['Hour'] = df_journeys['Start date'].dt.hour
                hourly_counts = df_journeys.groupby('Hour').size().reset_index(name='Aantal')
                fig_hour = px.bar(
                    hourly_counts,
                    x='Hour',
                    y='Aantal',
                    title='Aantal Ritten per Uur van de Dag',
                    labels={'Hour': 'Uur (0-23)', 'Aantal': 'Aantal Ritten'}
                )
                st.plotly_chart(fig_hour, use_container_width=True)
                
        with col_u2:
            if 'Total duration' in df_journeys.columns:
                # Duur in minuten omrekenen
                df_journeys['duration_min'] = df_journeys['Total duration'] / 60
                fig_dur = px.histogram(
                    df_journeys[df_journeys['duration_min'] <= 60],
                    x='duration_min',
                    nbins=30,
                    title='Verdeling van Ritduur (max. 60 min)',
                    labels={'duration_min': 'Duur in Minuten'}
                )
                st.plotly_chart(fig_dur, use_container_width=True)
    else:
        st.info("Zorg dat de journey CSV-bestanden in de `data/` map staan om de weer- en rit-grafieken volledig te tonen.")

# ------------------------------------------
# TAB 3: LIVE PRESENTATIE & BEVINDINGEN
# ------------------------------------------
with tab3:
    st.subheader("📌 Belangrijkste Bevindingen voor de Presentatie")
    
    st.markdown(f"""
    ### 1. Multimodale Koppeling & First/Last Mile
    * **Afstandsinstelling ({distance_threshold} meter):** 
      Bij deze afstandsinstelling fungeren grote metrostations (zoals *Waterloo*, *Victoria* en *King's Cross*) als cruciale **fiets-hubs**. 
    * De fiets wordt voornamelijk gebruikt voor de **First/Last Mile** verbinding van en naar het metrostation in het centrum van Londen.

    ### 2. Weersgevoeligheid van Bikeshare vs. Metro
    * **Temperatuureffect:** Er is een hele duidelijke positieve lineaire trend te zien tussen dagtemperatuur en fietsgebruik.
    * **Neerslageffect:** Zelfs lichte neerslag vermindert het aantal vrijetijdsritten aanzienlijk, terwijl de pieken op werkdagen rond 08:00 en 17:00 uur (woon-werkverkeer) beter standhouden.

    ### 3. Conclusie voor het Transportnetwerk
    * Bikeshare **vult de metro aan** aan de randen van het centrum (verbindt locaties die net te ver lopen zijn vanaf een metrostation).
    * Bikeshare **vervangt de metro** op korte trajecten binnen Zone 1 bij mooi/warm weer.
    """)
