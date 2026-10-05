import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_and_clean_weather(filepath='data/weather_london.csv'):
    """Inlezen en opschonen van het weerbestand."""
    df_weather = pd.read_csv(filepath)
    
    # 1. Hernoem de naamloze datumkolom
    if 'Unnamed: 0' in df_weather.columns:
        df_weather.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    
    # 2. Filter op het jaar 2022
    df_weather = df_weather[df_weather['date'].dt.year == 2022].copy()
    
    # 3. Verwijder vrijwel lege kolommen (zoals beschreven in de opdracht)
    empty_cols = [col for col in ['sun', 'snow', 'wgt'] if col in df_weather.columns]
    df_weather.drop(columns=empty_cols, inplace=True)
    
    return df_weather


def load_and_clean_metro(filepath='data/metrogebruik_londen_2022.csv'):
    """Inlezen en opschonen van het metrobestand."""
    df_metro = pd.read_csv(filepath)
    
    # 11 stations missen coördinaten (bijv. Bank and Monument, Edgware Road)
    # We vullen ontbrekende coördinaten aan met bekende locaties of verwijderen ze niet zomaar
    # Hier maken we een opzoektabel voor bekende gemiste stations
    missing_coords = {
        'Bank and Monument': (51.5133, -0.0890),
        'Edgware Road (Circle)': (51.5199, -0.1678),
        'Edgware Road (Bakerloo)': (51.5203, -0.1700),
        'Hammersmith (Dist&Picc)': (51.4924, -0.2237),
        'Hammersmith (H&C)': (51.4936, -0.2250)
    }
    
    for station, (lat, lon) in missing_coords.items():
        mask = df_metro['station_name'] == station if 'station_name' in df_metro.columns else df_metro.iloc[:, 0] == station
        df_metro.loc[mask, 'latitude'] = lat
        df_metro.loc[mask, 'longitude'] = lon
        
    # Drop eventuele overgebleven rijen zonder coördinaten
    df_metro = df_metro.dropna(subset=['latitude', 'longitude']).copy()
    
    return df_metro


def match_bikes_to_metro(bike_stations_df, metro_df, max_distance_m=250):
    """
    Koppel fietsstations aan het dichtstbijzijnde metrostation
    op basis van coördinaten (GeoPandas sjoin_nearest).
    """
    # Converteren naar GeoDataFrames (WGS84 -> EPSG:4326)
    metro_gdf = gpd.GeoDataFrame(
        metro_df,
        geometry=gpd.points_from_xy(metro_df['longitude'], metro_df['latitude']),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")  # Transformeer naar British National Grid (meters)

    bike_gdf = gpd.GeoDataFrame(
        bike_stations_df,
        geometry=gpd.points_from_xy(bike_stations_df['longitude'], bike_stations_df['latitude']),
        crs="EPSG:4326"
    ).to_crs("EPSG:27700")

    # Zoek dichtstbijzijnde metrostation per fietsstation
    joined = gpd.sjoin_nearest(bike_gdf, metro_gdf, distance_col="distance_m", how="left")
    
    # Filter op gekozen grens (bijv. 250 meter)
    nearby_bikes = joined[joined['distance_m'] <= max_distance_m].copy()
    
    return nearby_bikes
