import numpy as np
import pandas as pd
import requests

# We halen de locaties van de fietsstations op bij de TfL-API en koppelen elk fietsstation
# aan het dichtstbijzijnde metrostation. De afstand bewaren we, de grens kiezen we in het dashboard.

API = "https://api.tfl.gov.uk"

# Elf metrostations hebben geen coördinaat. Voor die stations zoeken we de locatie op via de
# StopPoint-zoekfunctie van TfL. Links staat de naam in ons bestand, rechts de zoekterm.
ZOEKTERMEN = {
    "Bank and Monument": "Bank",
    "Edgware Road (Bak)": "Edgware Road (Bakerloo)",
    "Edgware Road (DIS)": "Edgware Road (Circle Line)",
    "Hammersmith (DIS)": "Hammersmith (Dist&Picc Line)",
    "Hammersmith (H&C)": "Hammersmith (H&C Line)",
    "Heathrow Terminals 123 LU": "Heathrow Terminals 2 & 3",
    "Paddington TfL": "Paddington",
}


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


def zoek_locatie(term):
    r = requests.get(f"{API}/StopPoint/Search", params={"query": term,
                     "modes": "tube,dlr,overground,elizabeth-line"}, timeout=30)
    r.raise_for_status()
    treffers = r.json().get("matches", [])
    if not treffers:
        return None, None, None
    t = treffers[0]
    return t.get("lat"), t.get("lon"), t.get("name")


# Stap 1: fietsstations met locatie en aantal docks
r = requests.get(f"{API}/BikePoint", timeout=60)
r.raise_for_status()
rijen = []
for p in r.json():
    extra = {e["key"]: e["value"] for e in p.get("additionalProperties", [])}
    rijen.append({"station": p["commonName"].strip(), "lat": p["lat"], "lon": p["lon"],
                  "docks": pd.to_numeric(extra.get("NbDocks"), errors="coerce")})
bikepoints = pd.DataFrame(rijen)
print(f"BikePoint-API: {len(bikepoints)} fietsstations met locatie")

ritten = pd.read_csv("data/fiets_per_station.csv")
fiets = ritten.merge(bikepoints, on="station", how="left")
zonder = fiets[fiets["lat"].isna()].sort_values("starts", ascending=False)
print(f"Fietsstations uit de ritdata: {len(fiets)}, waarvan {len(zonder)} zonder locatie "
      f"({zonder['starts'].sum():,} starts, {zonder['starts'].sum() / fiets['starts'].sum():.2%} van alle ritten)")
print(zonder[["station", "starts"]].head(25).to_string(index=False))
fiets = fiets.dropna(subset=["lat"])

# Stap 2: ontbrekende metrocoördinaten aanvullen
metro = pd.read_csv("data/metrogebruik_londen_2022.csv")
metro["coord_bron"] = np.where(metro["lat"].notna(), "bestand", "")
print()
for i in metro.index[metro["lat"].isna()]:
    naam = metro.at[i, "station"]
    lat, lon, gevonden = zoek_locatie(ZOEKTERMEN.get(naam, naam))
    if lat is not None:
        metro.loc[i, ["lat", "lon", "coord_bron"]] = [lat, lon, "TfL StopPoint-API"]
    print(f"{naam:<28} [{metro.at[i, 'vervoerstype']}] -> {gevonden} ({lat}, {lon})")
metro.to_csv("data/metro_stations.csv", index=False)

# Stap 3: per fietsstation het dichtstbijzijnde metrostation en de afstand in meters
# Stations die per vervoerstype dubbel in het bestand staan hebben hun reizigers maar bij één rij; rijen met 0 reizigers slaan we over
m = metro[metro["lat"].notna() & (metro["totaal_per_jaar"] > 0)].reset_index(drop=True)
afstanden = haversine_m(fiets["lat"].to_numpy()[:, None], fiets["lon"].to_numpy()[:, None],
                        m["lat"].to_numpy()[None, :], m["lon"].to_numpy()[None, :])
dichtst = afstanden.argmin(axis=1)
fiets["metro_station"] = m.loc[dichtst, "station"].to_numpy()
fiets["metro_type"] = m.loc[dichtst, "vervoerstype"].to_numpy()
fiets["afstand_m"] = afstanden.min(axis=1).round(0)
fiets.to_csv("data/fietsstations.csv", index=False)

print()
for grens in [200, 250, 400, 500]:
    binnen = fiets[fiets["afstand_m"] <= grens]
    print(f"Binnen {grens} m: {len(binnen):>4} fietsstations, {binnen['metro_station'].nunique():>4} metrostations")
print("\nKlaar. Nieuwe bestanden: data/fietsstations.csv en data/metro_stations.csv")
