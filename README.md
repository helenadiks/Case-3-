# 🌍 Fiets en metro in Londen, 2022

Streamlit-dashboard voor case 3 (Introduction to Data Science en Visual Analytics, minor Data Science).

**Onderzoeksvraag:** wanneer en waar pakken Londenaren de deelfiets, en vervangt die de metro of vult hij hem aan?

## Wat er in het dashboard staat

| Tabblad | Vraag |
|---|---|
| 1. Wanneer fietst Londen? | Ritten per dag, week of maand, en het uurprofiel van stakingsdagen |
| 2. Weer en voorspelling | Effect van temperatuur en regen, en een voorspelmodel getoetst op november en december |
| 3. Waar staat de fiets? | Kaart van fietsstations en metrostations, en de koppeling tussen beide |
| Data en methode | Inspectie, opschonen, keuzes en bronnen |

## Draaien

```
pip install -r requirements.txt
streamlit run app.py
```

Het dashboard gebruikt alleen de samengevatte bestanden in de map `data`.

## Data opnieuw ophalen (optioneel)

De ruwe fietsdata (53 weekbestanden, ruim 11 miljoen ritten) staat niet in deze repository, want die is te groot.
De twee scripts halen alles opnieuw op en maken de samenvattingen:

```
python 01_fietsdata_ophalen.py      # ritten 2022 ophalen bij TfL en samenvatten
python 02_stations_koppelen.py      # locaties fietsstations ophalen en koppelen aan metrostations
```

## Bestanden

| Bestand | Inhoud |
|---|---|
| `app.py` | Het dashboard |
| `01_fietsdata_ophalen.py` | Haalt de ritdata 2022 op bij TfL en vat die samen per dag, uur, station en route |
| `02_stations_koppelen.py` | Haalt de fietsstations op via de TfL BikePoint-API en koppelt ze op afstand aan metrostations |
| `data/fiets_*.csv` | Samenvattingen van de ritdata |
| `data/fietsstations.csv` | Fietsstations met locatie, docks en dichtstbijzijnde metrostation |
| `data/metro_stations.csv` | Metrogebruik 2022 met aangevulde coördinaten |
| `data/metrogebruik_londen_2022.csv`, `data/weather_london.csv` | Aangeleverd via Brightspace |

## Bronnen

- Fietsritten: [TfL cycling data](https://cycling.data.tfl.gov.uk/), map usage-stats
- Fietsstations: [TfL Unified API, BikePoint](https://api.tfl.gov.uk/BikePoint)
- Metrogebruik en weer: aangeleverd via Brightspace
- Stakingsdagen: [Wikipedia, London Underground strikes](https://en.wikipedia.org/wiki/London_Underground_strikes)
