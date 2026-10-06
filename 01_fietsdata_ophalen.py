import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

# We halen alle ritbestanden van 2022 op bij TfL en vatten ze samen tot kleine bestanden.
# De ruwe downloads blijven lokaal in data/raw en gaan niet naar GitHub.

BUCKET_LIJST = "https://s3-eu-west-1.amazonaws.com/cycling.data.tfl.gov.uk/?list-type=2&prefix=usage-stats/"
DOWNLOAD_BASIS = "https://cycling.data.tfl.gov.uk/"
JAAR = 2022
MIN_DUUR_S = 60          # korter dan een minuut zien we als een mislukte of geannuleerde rit
MAX_DUUR_S = 24 * 3600   # langer dan een dag zien we als een niet goed teruggezette fiets

RAW = Path("data/raw")
OUT = Path("data")
RAW.mkdir(parents=True, exist_ok=True)

TEST = "--test" in sys.argv


def bestandenlijst():
    r = requests.get(BUCKET_LIJST, timeout=60)
    r.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [k.text for k in ET.fromstring(r.content).findall(".//s3:Key", ns)]
    keys = [k for k in keys if k.lower().endswith(".csv") and str(JAAR) in k]
    return sorted(keys)


def download(key):
    pad = RAW / Path(key).name
    if pad.exists() and pad.stat().st_size > 0:
        return pad
    url = DOWNLOAD_BASIS + quote(key)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(pad, "wb") as f:
            for blok in r.iter_content(chunk_size=1 << 20):
                f.write(blok)
    return pad


def inlezen(pad):
    # TfL heeft in 2022 de layout veranderd, dus we zetten beide varianten om naar dezelfde kolommen
    df = pd.read_csv(pad, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    if "rental id" in df.columns:
        uit = pd.DataFrame({
            "rit_id": df["rental id"],
            "start": pd.to_datetime(df["start date"], dayfirst=True, errors="coerce"),
            "startstation": df["startstation name"],
            "eindstation": df["endstation name"],
            "duur_s": pd.to_numeric(df["duration"], errors="coerce"),
        })
        layout = "oud"
    elif "number" in df.columns:
        uit = pd.DataFrame({
            "rit_id": df["number"],
            "start": pd.to_datetime(df["start date"], errors="coerce"),
            "startstation": df["start station"],
            "eindstation": df["end station"],
            "duur_s": pd.to_numeric(df["total duration (ms)"], errors="coerce") / 1000,
        })
        layout = "nieuw"
    else:
        raise ValueError(f"Onbekende layout in {pad.name}: {list(df.columns)}")

    uit["startstation"] = uit["startstation"].astype(str).str.strip()
    uit["eindstation"] = uit["eindstation"].astype(str).str.strip()
    return uit, layout


keys = bestandenlijst()
print(f"{len(keys)} bestanden gevonden met {JAAR} in de naam")
if TEST:
    keys = keys[:2]
    print("Testmodus: alleen de eerste 2 bestanden")

per_dag, per_uur, starts, eindes, routes = [], [], [], [], []
ids = []
log = []

for i, key in enumerate(keys, 1):
    print(f"[{i}/{len(keys)}] {Path(key).name}", flush=True)
    pad = download(key)
    df, layout = inlezen(pad)

    n_bestand = len(df)
    df = df[df["start"].dt.year == JAAR]
    n_jaar = len(df)

    geen_duur = df["duur_s"].isna()
    te_kort = df["duur_s"] < MIN_DUUR_S
    te_lang = df["duur_s"] > MAX_DUUR_S
    df["dag"] = df["start"].dt.normalize()

    # Per dag houden we bij hoeveel er ruw was en hoeveel we weggooien, voor het opschoon-verhaal
    ruw = df.groupby("dag").size().rename("ritten_ruw")
    kort = df[te_kort].groupby("dag").size().rename("weg_te_kort")
    lang = df[te_lang].groupby("dag").size().rename("weg_te_lang")
    leeg = df[geen_duur].groupby("dag").size().rename("weg_geen_duur")

    schoon = df[~(geen_duur | te_kort | te_lang)].copy()
    ids.append(schoon["rit_id"].to_numpy(dtype="int64", na_value=-1))

    dag = schoon.groupby("dag").agg(ritten=("rit_id", "size"), duur_totaal_s=("duur_s", "sum"))
    per_dag.append(pd.concat([ruw, kort, lang, leeg, dag], axis=1).fillna(0))

    schoon["uur"] = schoon["start"].dt.hour
    per_uur.append(schoon.groupby(["dag", "uur"]).size().rename("ritten"))
    starts.append(schoon.groupby("startstation").size().rename("starts"))
    eindes.append(schoon.groupby("eindstation").size().rename("eindes"))
    routes.append(schoon.groupby(["startstation", "eindstation"]).size().rename("ritten"))

    log.append({"bestand": Path(key).name, "layout": layout, "rijen_bestand": n_bestand,
                "rijen_in_2022": n_jaar, "rijen_schoon": len(schoon)})

# Bestanden kunnen elkaar op de randen overlappen, dus we checken op dubbele rit-ID's
alle_ids = np.concatenate(ids)
alle_ids = alle_ids[alle_ids >= 0]
uniek, aantallen = np.unique(alle_ids, return_counts=True)
n_dubbel = int((aantallen - 1).sum())

dag = pd.concat(per_dag).groupby(level=0).sum().sort_index()
dag.index.name = "datum"
dag["gem_duur_min"] = (dag["duur_totaal_s"] / dag["ritten"] / 60).round(2)
dag = dag.drop(columns="duur_totaal_s").astype({c: "int64" for c in
      ["ritten_ruw", "weg_te_kort", "weg_te_lang", "weg_geen_duur", "ritten"]})
dag.to_csv(OUT / "fiets_per_dag.csv")

uur = pd.concat(per_uur).groupby(level=[0, 1]).sum().reset_index()
uur = uur.rename(columns={"dag": "datum"})
uur.to_csv(OUT / "fiets_per_uur.csv", index=False)

station = pd.concat([pd.concat(starts).groupby(level=0).sum(),
                     pd.concat(eindes).groupby(level=0).sum()], axis=1).fillna(0).astype("int64")
station.index.name = "station"
station.sort_values("starts", ascending=False).to_csv(OUT / "fiets_per_station.csv")

route = pd.concat(routes).groupby(level=[0, 1]).sum().sort_values(ascending=False).reset_index()
route["rondrit"] = route["startstation"] == route["eindstation"]
route.head(2000).to_csv(OUT / "fiets_routes_top.csv", index=False)

pd.DataFrame(log).to_csv(OUT / "fiets_inspectie_bestanden.csv", index=False)

print()
print("Klaar.")
print(f"Ritten in {JAAR} ruw:          {int(dag['ritten_ruw'].sum()):>12,}")
print(f"Weg, korter dan 1 minuut:   {int(dag['weg_te_kort'].sum()):>12,}")
print(f"Weg, langer dan 24 uur:     {int(dag['weg_te_lang'].sum()):>12,}")
print(f"Weg, geen duur:             {int(dag['weg_geen_duur'].sum()):>12,}")
print(f"Over na opschonen:          {int(dag['ritten'].sum()):>12,}")
print(f"Dubbele rit-ID's:           {n_dubbel:>12,}")
print(f"Dagen in de reeks:          {len(dag):>12}")
print(f"Fietsstations:              {len(station):>12}")
print(f"Layouts: {pd.DataFrame(log)['layout'].value_counts().to_dict()}")