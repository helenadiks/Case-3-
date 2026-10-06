from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fiets en metro in Londen", page_icon="🚲", layout="wide")

DATA = Path(__file__).parent / "data"

# Vaste kleuren, zodat een reeks overal dezelfde kleur houdt
BLAUW = "#2a78d6"
ORANJE = "#eb6834"
GROEN = "#1baf7a"
GRIJS = "#898781"
ROOD = "#d03b3b"
BLAUWE_SCHAAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Metrostakingen in 2022 (bronnen: zie tabblad Data en methode)
STAKINGEN = {
    "2022-03-01": "Metrostaking",
    "2022-03-03": "Metrostaking",
    "2022-06-06": "Metrostaking (deel van het net)",
    "2022-06-21": "Metro- en treinstaking",
    "2022-06-22": "Dag na metrostaking, dienst nog beperkt",
    "2022-08-19": "Metrostaking",
    "2022-11-10": "Metrostaking",
}

# Britse feestdagen 2022, inclusief de extra dagen voor het jubileum en de uitvaart van de koningin
FEESTDAGEN = ["2022-01-03", "2022-04-15", "2022-04-18", "2022-05-02", "2022-06-02",
              "2022-06-03", "2022-08-29", "2022-09-19", "2022-12-26", "2022-12-27"]

# Op dit weekend stapte TfL over op een nieuw systeem, er zijn bijna geen ritten geregistreerd
SYSTEEMWISSEL = ["2022-09-10", "2022-09-11"]


def nl(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


MAANDEN = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]
DAGEN = ["ma", "di", "wo", "do", "vr", "za", "zo"]


def nl_datum(d):
    return f"{DAGEN[d.dayofweek]} {d.day} {MAANDEN[d.month - 1]}"


def maand_as(fig):
    starts = pd.date_range("2022-01-01", "2022-12-01", freq="MS")
    fig.update_xaxes(tickvals=starts, ticktext=MAANDEN)
    return fig


def opmaak(fig, hoogte=420):
    fig.update_layout(height=hoogte, margin=dict(l=10, r=10, t=50, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                      hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=False)
    return fig


def daglengte_uren(datum, breedte=51.5):
    # Daglengte uit de zonnedeclinatie, standaardformule voor zonsopkomst en zonsondergang
    dag = datum.dt.dayofyear.to_numpy()
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + dag) / 365)
    cos_h = -np.tan(np.radians(breedte)) * np.tan(decl)
    return 24 / np.pi * np.arccos(np.clip(cos_h, -1, 1))


@st.cache_data
def laad_dagen():
    dag = pd.read_csv(DATA / "fiets_per_dag.csv", parse_dates=["datum"])
    weer = pd.read_csv(DATA / "weather_london.csv").rename(columns={"Unnamed: 0": "datum"})
    weer["datum"] = pd.to_datetime(weer["datum"])
    df = dag.merge(weer[["datum", "tavg", "tmin", "tmax", "prcp", "wspd"]], on="datum", how="left")

    df["systeemwissel"] = df["datum"].isin(pd.to_datetime(SYSTEEMWISSEL))
    df.loc[df["systeemwissel"], ["ritten", "gem_duur_min"]] = np.nan
    df["staking"] = df["datum"].dt.strftime("%Y-%m-%d").map(STAKINGEN)
    df["feestdag"] = df["datum"].isin(pd.to_datetime(FEESTDAGEN))
    df["vrij"] = (df["datum"].dt.dayofweek >= 5) | df["feestdag"]
    df["soort"] = np.where(df["vrij"], "Weekend en feestdag", "Werkdag")
    df["daglengte"] = daglengte_uren(df["datum"])
    # Neerslag is scheef verdeeld (meestal 0, soms 27 mm), daarom gebruiken we in het model log(1 + mm)
    df["log_regen"] = np.log1p(df["prcp"])
    df["nieuw_systeem"] = df["datum"] >= "2022-09-12"
    df["regen"] = pd.cut(df["prcp"], [-0.1, 0.2, 2, 100], labels=["Droog", "Licht (0,2 tot 2 mm)", "Flink (meer dan 2 mm)"])
    return df


@st.cache_data
def laad_uren():
    uur = pd.read_csv(DATA / "fiets_per_uur.csv", parse_dates=["datum"])
    uur = uur[~uur["datum"].isin(pd.to_datetime(SYSTEEMWISSEL))]
    return uur


@st.cache_data
def laad_stations():
    fiets = pd.read_csv(DATA / "fietsstations.csv")
    metro = pd.read_csv(DATA / "metro_stations.csv")
    metro["zone_nr"] = pd.to_numeric(metro["zone"].astype(str).str.split(",").str[0], errors="coerce")
    alle = pd.read_csv(DATA / "fiets_per_station.csv")
    routes = pd.read_csv(DATA / "fiets_routes_top.csv")
    bestanden = pd.read_csv(DATA / "fiets_inspectie_bestanden.csv")
    weer = pd.read_csv(DATA / "weather_london.csv")
    return fiets, metro, alle, routes, bestanden, weer


@st.cache_data
def model_fit(df, zonder_stakingen):
    # Lineaire regressie met de kleinste-kwadratenmethode: ritten per dag uit weer, vrije dag en daglengte.
    # We trainen op januari tot en met oktober en toetsen op november en december.
    kenmerken = ["tmax", "log_regen", "wspd", "vrij", "daglengte", "nieuw_systeem"]
    d = df.dropna(subset=["ritten"] + kenmerken).copy()
    d[["vrij", "nieuw_systeem"]] = d[["vrij", "nieuw_systeem"]].astype(float)
    train = d["datum"] < "2022-11-01"
    fit = train & (d["staking"].isna() if zonder_stakingen else True)

    X = np.column_stack([np.ones(len(d)), d[kenmerken].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(X[fit.to_numpy()], d.loc[fit, "ritten"].to_numpy(), rcond=None)
    d["voorspeld"] = X @ beta
    rest = d.loc[fit, "ritten"] - d.loc[fit, "voorspeld"]
    sd = rest.std()
    d["onder"] = d["voorspeld"] - 1.96 * sd
    d["boven"] = d["voorspeld"] + 1.96 * sd
    d["fout"] = d["ritten"] - d["voorspeld"]
    d["set"] = np.where(train, "Training (jan t/m okt)", "Toets (nov en dec)")
    coef = pd.Series(beta[1:], index=["Max. temperatuur (per °C)", "Neerslag (per stap in log(1 + mm))", "Wind (per km/u)",
                                      "Vrije dag", "Daglengte (per uur)", "Nieuw TfL-systeem (vanaf 12 sep)"])
    return d, coef, sd



def metro_index(fiets, metro, grens):
    # Per metrostation tellen we de fietsdocks binnen de grens en vergelijken we het metrogebruik
    # met de mediaan van stations in dezelfde zone (verwacht gebruik op grond van de locatie)
    binnen = fiets[fiets["afstand_m"] <= grens]
    per_metro = binnen.groupby(["metro_station", "metro_type"]).agg(docks=("docks", "sum"), fietsstations=("station", "size")).reset_index()
    m = metro.merge(per_metro, left_on=["station", "vervoerstype"], right_on=["metro_station", "metro_type"], how="left")
    m[["docks", "fietsstations"]] = m[["docks", "fietsstations"]].fillna(0)
    # Alleen zone 1 en 2: daar staan vrijwel alle fietsstations, dus daar is de vergelijking eerlijk
    m = m[(m["zone_nr"].isin([1, 2])) & (m["totaal_per_jaar"] > 0)].copy()
    m["index"] = m["totaal_per_jaar"] / m.groupby("zone_nr")["totaal_per_jaar"].transform("median")
    m["groep"] = pd.cut(m["docks"], [-1, 40, 1000], labels=["40 fietsdocks of minder", "Meer dan 40 fietsdocks"])
    return m, binnen


df = laad_dagen()
uur = laad_uren()
fiets, metro, alle_stations, routes, bestanden, weer_ruw = laad_stations()

normaal_werkdag = df.loc[(df["soort"] == "Werkdag") & df["staking"].isna(), "ritten"].median()
stakingsdagen = df[df["staking"].notna()].sort_values("ritten", ascending=False)
top4_staking = df.nlargest(4, "ritten")["staking"].notna().sum()
werk = df.dropna(subset=["ritten"])
werk = werk[(werk["soort"] == "Werkdag") & werk["staking"].isna()]
droog_med = werk.loc[werk["regen"] == "Droog", "ritten"].median()
nat_med = werk.loc[werk["regen"] == "Flink (meer dan 2 mm)", "ritten"].median()
m250, _ = metro_index(fiets, metro, 250)
index_veel = m250.loc[m250["groep"] == "Meer dan 40 fietsdocks", "index"].median()

# Eerste laag: de vraag en het antwoord in één oogopslag
st.title("Fiets en metro in Londen, 2022")
st.markdown(
    "#### Wanneer en waar pakken Londenaren de deelfiets, en vervangt die de metro of vult hij hem aan?\n"
    "**Londen fietst op warme, droge werkdagen en de fiets vult de metro aan. Bij een metrostaking springt de fiets in, "
    "maar vangt hij maar een klein deel van de reizigers op.**"
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Fietsritten in 2022", f"{nl(df['ritten'].sum() / 1e6, 1)} mln", help="Na opschonen, zie Data en methode.")
k2.metric("Op een stakingsdag", f"+{nl((stakingsdagen['ritten'].median() / normaal_werkdag - 1) * 100)}%",
          help=f"Mediaan {nl(stakingsdagen['ritten'].median())} ritten tegenover {nl(normaal_werkdag)} op een gewone werkdag.")
k3.metric("Bij flinke regen (werkdag)", f"{nl((nat_med / droog_med - 1) * 100)}%",
          help=f"Mediaan {nl(nat_med)} ritten bij meer dan 2 mm regen, {nl(droog_med)} op een droge werkdag.")
k4.metric("Metro met veel fietsdocks ernaast", f"{nl(index_veel, 1)}× zo druk",
          help="Metrostations in zone 1 en 2 met meer dan 40 fietsdocks binnen 250 m, vergeleken met de mediaan van hun zone.")

tab1, tab2, tab3, tab4 = st.tabs(["1. Wanneer fietst Londen?", "2. Weer en voorspelling", "3. Waar staat de fiets?", "Data en methode"])

# Tab 1: lijngrafiek, met het uurprofiel als tweede laag
with tab1:
    c1, c2 = st.columns([1, 2])
    eenheid = c1.radio("Tijdseenheid", ["Dag", "Week", "Maand"], horizontal=True,
                       help="Per dag zie je de pieken, per week en maand het seizoen.")
    if eenheid == "Dag":
        reeksen = c2.multiselect("Reeksen", ["Elke dag", "7-daags gemiddelde"], default=["Elke dag", "7-daags gemiddelde"])
    else:
        reeksen = c2.multiselect("Reeksen", ["Alle dagen", "Werkdag", "Weekend en feestdag"], default=["Werkdag", "Weekend en feestdag"])
    kleuren = {"Elke dag": "#86b6ef", "7-daags gemiddelde": BLAUW, "Alle dagen": GRIJS, "Werkdag": BLAUW, "Weekend en feestdag": GROEN}

    fig = go.Figure()
    reeks = df.set_index("datum")["ritten"]
    if eenheid == "Dag":
        if "Elke dag" in reeksen:
            fig.add_trace(go.Scatter(x=reeks.index, y=reeks.values, name="Elke dag", mode="lines+markers", connectgaps=False,
                                     line=dict(color=kleuren["Elke dag"], width=1), marker=dict(size=3),
                                     hovertemplate="%{x|%d-%m-%Y}<br>%{y:,.0f} ritten<extra></extra>"))
        if "7-daags gemiddelde" in reeksen:
            gem = reeks.rolling(7, center=True, min_periods=5).mean()
            fig.add_trace(go.Scatter(x=gem.index, y=gem.values, name="7-daags gemiddelde", mode="lines",
                                     line=dict(color=BLAUW, width=2.5), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=stakingsdagen["datum"], y=stakingsdagen["ritten"], mode="markers", name="Metrostaking (klik)",
                                 marker=dict(symbol="star", size=15, color=ROOD, line=dict(color="white", width=1)),
                                 text=stakingsdagen["staking"],
                                 hovertemplate="%{x|%d-%m-%Y}<br>%{y:,.0f} ritten<br>%{text}<extra></extra>"))
        titel = f"Fietsritten per dag: de {top4_staking} hoogste pieken zijn metrostakingen"
    else:
        regel = {"Week": "W-SUN", "Maand": "MS"}[eenheid]
        for r in reeksen:
            sub = df if r == "Alle dagen" else df[df["soort"] == r]
            s = sub.set_index("datum")["ritten"].resample(regel).mean()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=r, mode="lines+markers", connectgaps=False,
                                     line=dict(color=kleuren[r], width=2.5), marker=dict(size=6),
                                     hovertemplate="%{x|%d-%m-%Y}<br>gemiddeld %{y:,.0f} ritten per dag<extra>" + r + "</extra>"))
        titel = f"Gemiddeld aantal ritten per dag, per {eenheid.lower()}: zomer twee keer zo druk als winter"
    fig.add_vrect(x0="2022-09-09 12:00", x1="2022-09-11 12:00", fillcolor=GRIJS, opacity=0.25, line_width=0)
    fig.add_annotation(x="2022-09-10", y=1, yref="paper", text="Systeemwissel TfL", showarrow=False,
                       font=dict(size=11, color=GRIJS), yanchor="bottom")
    fig.update_layout(title=titel, yaxis_title="Ritten per dag", xaxis_title=None, clickmode="event+select")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    maand_as(fig)
    klik = st.plotly_chart(opmaak(fig, 470), width="stretch", on_select="rerun", selection_mode="points", key="lijn")
    with st.expander("Toelichting bij deze grafiek"):
        st.markdown(
            "- **Per dag** als standaard, omdat een staking één dag duurt en in een weekgemiddelde verdwijnt. Het 7-daags gemiddelde laat het seizoen zien.\n"
            "- Bij **week en maand** tonen we het gemiddelde per dag, zodat een korte week of maand niet lager lijkt.\n"
            "- **10 en 11 september** laten we leeg (systeemwissel TfL) in plaats van ze door te verbinden.\n"
            "- Inzoomen: sleep in de balk onder de grafiek."
        )

    # Tweede laag: wat gebeurde er op die piekdag? Klik op een dag in de grafiek of kies hieronder.
    gekozen = None
    punten = klik.selection.points if klik and klik.selection else []
    if punten and eenheid == "Dag":
        gekozen = pd.Timestamp(str(punten[0]["x"])[:10])
    opties = {f"{nl_datum(pd.Timestamp(d))}, {s.lower()}": d for d, s in STAKINGEN.items()}
    c1, c2 = st.columns([1, 3])
    if gekozen is None:
        keuze = c1.selectbox("Klik op een dag in de grafiek, of kies een stakingsdag", list(opties.keys()), index=5)
        gekozen = pd.Timestamp(opties[keuze])
    else:
        c1.markdown(f"**Gekozen in de grafiek:** {nl_datum(gekozen)}")
        c1.caption("Klik naast de punten om de selectie op te heffen.")

    werk_dagen = df.loc[(df["soort"] == "Werkdag") & df["staking"].isna(), "datum"]
    prof_werk = uur[uur["datum"].isin(werk_dagen)].groupby("uur")["ritten"].mean()
    prof_weekend = uur[uur["datum"].isin(df.loc[df["vrij"], "datum"])].groupby("uur")["ritten"].mean()
    prof_dag = uur[uur["datum"] == gekozen].set_index("uur")["ritten"].reindex(range(24), fill_value=0)
    vergelijk = prof_weekend if df.loc[df["datum"] == gekozen, "vrij"].any() else prof_werk
    extra = (prof_dag.sum() / vergelijk.sum() - 1) * 100

    is_vrij = vergelijk is prof_weekend
    verschil = prof_dag - vergelijk
    extra_ritten = verschil.sum()
    piek_uur = int(verschil.idxmax())

    # Context dun en licht, de gekozen dag dik; het gekleurde vlak is het verschil met een gewone dag
    fig = go.Figure()
    andere = prof_werk if is_vrij else prof_weekend
    fig.add_trace(go.Scatter(x=andere.index, y=andere.values, name="Gemiddeld weekend" if not is_vrij else "Gemiddelde werkdag",
                             mode="lines", line=dict(color="#c3c2b7", width=1.5, dash="dot"),
                             hovertemplate="%{x}:00 uur<br>%{y:,.0f} ritten<extra></extra>"))
    fig.add_trace(go.Scatter(x=vergelijk.index, y=vergelijk.values, name="Gemiddelde " + ("weekenddag" if is_vrij else "werkdag"),
                             mode="lines", line=dict(color=GROEN if is_vrij else BLAUW, width=2),
                             hovertemplate="%{x}:00 uur<br>%{y:,.0f} ritten<extra>Gemiddeld</extra>"))
    fig.add_trace(go.Scatter(x=prof_dag.index, y=prof_dag.values, name=nl_datum(gekozen), mode="lines", fill="tonexty",
                             fillcolor="rgba(208,59,59,0.15)", line=dict(color=ROOD, width=3),
                             hovertemplate="%{x}:00 uur<br>%{y:,.0f} ritten<extra>" + nl_datum(gekozen) + "</extra>"))
    fig.update_layout(title=f"{nl_datum(gekozen)}: {'+' if extra_ritten >= 0 else ''}{nl(extra_ritten)} fietsritten ({'+' if extra >= 0 else ''}"
                            f"{nl(extra)}%) t.o.v. een gemiddelde {'weekenddag' if is_vrij else 'werkdag'}, grootste verschil rond {piek_uur}:00",
                      xaxis_title="Uur van de dag", yaxis_title="Ritten per uur")
    fig.update_xaxes(dtick=2)
    c2.plotly_chart(opmaak(fig, 380), width="stretch")
    c1.caption("Het rode vlak zijn de extra fietsritten ten opzichte van een gewone dag.")

# Tab 2: weer en voorspelling
with tab2:
    st.subheader("Hoe warmer en droger, hoe meer fietsers, en dat is goed te voorspellen")
    c1, c2 = st.columns([3, 1])
    soort = c2.radio("Welke dagen", ["Werkdag", "Weekend en feestdag"], key="weer_soort",
                     help="Stakingsdagen laten we hier weg: dat is geen weer. Klik in de legenda om een regenklasse aan of uit te zetten.")
    d = df.dropna(subset=["ritten"])
    d = d[(d["soort"] == soort) & d["staking"].isna()]
    b, a = np.polyfit(d["tmax"], d["ritten"], 1)
    r = np.corrcoef(d["tmax"], d["ritten"])[0, 1]
    regen_med = d.groupby("regen", observed=True)["ritten"].median()

    fig = go.Figure()
    for regen, kleur in zip(d["regen"].cat.categories, ["#9ec5f4", "#3987e5", "#0d366b"]):
        s = d[d["regen"] == regen]
        fig.add_trace(go.Scatter(x=s["tmax"], y=s["ritten"], mode="markers", name=regen,
                                 marker=dict(color=kleur, size=8, opacity=0.85, line=dict(color="white", width=1)),
                                 customdata=s[["datum", "prcp"]],
                                 hovertemplate="%{customdata[0]|%d-%m-%Y}<br>%{x} °C, %{customdata[1]} mm<br>%{y:,.0f} ritten<extra></extra>"))
    xs = np.array([d["tmax"].min(), d["tmax"].max()])
    fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name=f"Trend (r = {nl(r, 2)})",
                             line=dict(color=GRIJS, dash="dash", width=2), hoverinfo="skip"))
    fig.update_layout(title=f"Elke graad warmer: ongeveer {nl(b)} ritten per dag meer",
                      xaxis_title="Maximumtemperatuur (°C)", yaxis_title="Ritten per dag")
    c1.plotly_chart(opmaak(fig, 400), width="stretch")
    c2.metric("Droog (mediaan)", nl(regen_med.get("Droog", np.nan)))
    c2.metric("Flinke regen (mediaan)", nl(regen_med.get("Flink (meer dan 2 mm)", np.nan)),
              f"{nl((regen_med.get('Flink (meer dan 2 mm)') / regen_med.get('Droog') - 1) * 100)}%")

    st.divider()
    c1, c2 = st.columns([3, 1])
    zonder = c2.toggle("Stakingsdagen uitsluiten bij het trainen", value=True,
                       help="Een staking is geen weer. Zet hem uit om te zien of het model daar gevoelig voor is.")
    res, coef, sd = model_fit(df, zonder)
    toets = res[res["set"].str.startswith("Toets")]
    train = res[res["set"].str.startswith("Training")]
    mae_toets = toets["fout"].abs().mean()
    mae_train = train.loc[train["staking"].isna(), "fout"].abs().mean()
    binnen_band = ((toets["ritten"] >= toets["onder"]) & (toets["ritten"] <= toets["boven"])).mean()
    c2.metric("Gemiddelde fout, nov en dec", f"{nl(mae_toets)} ritten",
              help=f"Dagen die het model nooit gezien heeft. Gemiddeld waren er {nl(toets['ritten'].mean())} ritten per dag. "
                   f"Op de trainingsdagen (zonder stakingen) was de fout {nl(mae_train)} ritten.")
    c2.metric("Nov en dec binnen de bandbreedte", f"{nl(binnen_band * 100)}%")

    heel_jaar = c2.toggle("Heel 2022 tonen", value=False, help="Standaard zie je de toetsperiode met september en oktober als context.")
    kal = res.set_index("datum").reindex(df["datum"])
    tr_k = kal[kal.index < "2022-11-01"]
    te_k = kal[kal.index >= "2022-11-01"]

    fig = go.Figure()
    # Trainingsperiode als context: dun en grijs
    fig.add_trace(go.Scatter(x=tr_k.index, y=tr_k["ritten"], name="Werkelijk (training)", mode="lines", connectgaps=False,
                             line=dict(color="#c3c2b7", width=1.2), hovertemplate="%{x|%d-%m-%Y}<br>werkelijk %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tr_k.index, y=tr_k["voorspeld"], name="Voorspeld (training)", mode="lines",
                             line=dict(color="#9ec5f4", width=1.2), hovertemplate="%{x|%d-%m-%Y}<br>voorspeld %{y:,.0f}<extra></extra>"))
    # Toetsperiode als highlight: bandbreedte, voorspelling en werkelijkheid
    fig.add_trace(go.Scatter(x=list(te_k.index) + list(te_k.index[::-1]), y=list(te_k["boven"]) + list(te_k["onder"][::-1]),
                             fill="toself", fillcolor="rgba(42,120,214,0.18)", line=dict(width=0), name="95% bandbreedte", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=te_k.index, y=te_k["voorspeld"], name="Voorspeld", mode="lines", line=dict(color=BLAUW, width=2.5),
                             hovertemplate="%{x|%d-%m-%Y}<br>voorspeld %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=te_k.index, y=te_k["ritten"], name="Werkelijk", mode="lines+markers", line=dict(color="#0b0b0b", width=2),
                             marker=dict(size=5), hovertemplate="%{x|%d-%m-%Y}<br>werkelijk %{y:,.0f}<extra></extra>"))
    # De twee grootste missers in de toetsperiode labelen we in de grafiek zelf
    for _, rij in toets.reindex(toets["fout"].abs().sort_values(ascending=False).index).head(2).iterrows():
        uitleg = rij["staking"].lower() if pd.notna(rij["staking"]) else ("kerst" if rij["datum"].month == 12 and rij["datum"].day >= 20 else "geen verklaring")
        fig.add_annotation(x=rij["datum"], y=rij["ritten"], text=f"{nl_datum(rij['datum'])}: {uitleg}<br>{'+' if rij['fout'] > 0 else ''}{nl(rij['fout'])} ritten",
                           showarrow=True, arrowhead=0, ax=0, ay=-40 if rij["fout"] > 0 else 40, font=dict(size=11, color=ROOD),
                           bgcolor="rgba(255,255,255,0.85)")
    fig.add_vline(x="2022-11-01", line_dash="dot", line_color=GRIJS)
    fig.add_annotation(x="2022-11-01", y=0.02, yref="paper", text=" Toetsperiode", showarrow=False, xanchor="left",
                       font=dict(size=11, color="#52514e"), yanchor="bottom")
    fig.update_layout(title=f"Getoetst op nov en dec: {nl(binnen_band * 100)}% van de dagen valt binnen de bandbreedte, de staking niet",
                      yaxis_title="Ritten per dag", yaxis_range=[0, 75000 if heel_jaar else 56000])
    if not heel_jaar:
        fig.update_xaxes(range=["2022-09-01", "2022-12-31"])
        fig.update_xaxes(tickvals=pd.date_range("2022-09-01", "2022-12-31", freq="MS"), ticktext=["sep", "okt", "nov", "dec"])
    else:
        maand_as(fig)
    fig = opmaak(fig, 470)
    fig.update_layout(legend=dict(orientation="h", y=-0.1, yanchor="top", x=0))
    c1.plotly_chart(fig, width="stretch")
    with st.expander("Toelichting: hoe werkt het model en wanneer klopt het niet?"):
      st.markdown(
        "**Hoe.** Lineaire regressie op temperatuur, regen, wind, vrije dag en daglengte. Getraind op januari tot en met oktober, "
        "getoetst op november en december.  \n"
        "**Waar het breekt.** Een metrostaking ziet het model niet aankomen, daar zit het 15.000 tot 30.000 ritten te laag. "
        "Ook de kerstweek is lager dan voorspeld.  \n"
        "**Aanname.** Het verband tussen weer en fietsen blijft gelijk. Redelijk voor dagen tot een paar maanden vooruit; "
        "over jaren niet, want het aantal stations, de prijzen en het aanbod van e-bikes veranderen."
    )

# Tab 3: kaart en koppeling fiets en metro
with tab3:
    st.subheader("De fiets staat waar de metro al het drukst is")
    c1, c2 = st.columns([1, 3])
    maat = c1.radio("Kleur de vakken naar", ["Aantal fietsstations", "Fietsritten (start)", "Aantal docks"], key="maat",
                    help="Vakken van 1 bij 1 km. De kleur is in vijf klassen met elk evenveel vakken (kwantielen), "
                         "want een paar vakken rond Hyde Park en Waterloo zouden anders alle kleur opeisen.")
    toon_metro = c1.toggle("Metrostations tonen", value=True)
    toon_fiets = c1.toggle("Losse fietsstations tonen", value=False)

    # We verdelen de stad in vakken van ongeveer 1 bij 1 km en tellen per vak
    DLAT, DLON = 0.009, 0.0144
    f = fiets.copy()
    f["rij"] = np.floor(f["lat"] / DLAT).astype(int)
    f["kol"] = np.floor(f["lon"] / DLON).astype(int)
    f["wijk"] = f["station"].str.split(",").str[-1].str.strip()
    f["vak"] = f["rij"].astype(str) + "_" + f["kol"].astype(str)
    kolom = {"Aantal fietsstations": "station", "Fietsritten (start)": "starts", "Aantal docks": "docks"}[maat]
    agg = ("station", "size") if kolom == "station" else (kolom, "sum")
    vakken = f.groupby(["vak", "rij", "kol"]).agg(waarde=agg, stations=("station", "size"),
                                                   ritten=("starts", "sum"), docks=("docks", "sum"),
                                                   wijk=("wijk", lambda s: s.value_counts().index[0])).reset_index()
    vakken = vakken.sort_values("waarde", ascending=False).reset_index(drop=True)
    vakken["naam"] = vakken["wijk"] + " (nr. " + (vakken.index + 1).astype(str) + ")"

    # Kwantielklassen: elke klasse bevat ongeveer evenveel vakken, zo eist één druk gebied niet alle kleur op
    # Bij kleine aantallen vallen grenzen soms samen, dan houden we minder dan vijf klassen over
    vakken["klasse"] = pd.qcut(vakken["waarde"], 5, labels=False, duplicates="drop")
    n_klassen = int(vakken["klasse"].max()) + 1
    randen = vakken.groupby("klasse")["waarde"].agg(["min", "max"])
    labels = [nl(lo) if lo == hi else f"{nl(lo)} tot {nl(hi)}" for lo, hi in zip(randen["min"], randen["max"])]
    kleuren5 = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    kleuren_k = [kleuren5[round(i * 4 / max(n_klassen - 1, 1))] for i in range(n_klassen)]
    schaal = []
    for i, k in enumerate(kleuren_k):
        schaal += [[i / n_klassen, k], [(i + 1) / n_klassen, k]]

    features = [{"type": "Feature", "id": v["vak"], "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [[[v["kol"] * DLON, v["rij"] * DLAT], [(v["kol"] + 1) * DLON, v["rij"] * DLAT],
                                                                  [(v["kol"] + 1) * DLON, (v["rij"] + 1) * DLAT],
                                                                  [v["kol"] * DLON, (v["rij"] + 1) * DLAT], [v["kol"] * DLON, v["rij"] * DLAT]]]}}
                for _, v in vakken.iterrows()]
    geo = {"type": "FeatureCollection", "features": features}

    namen = vakken.sort_values("waarde", ascending=False)["naam"].tolist()
    keuze_vak = c1.selectbox("Kies een gebied", ["Geen"] + namen,
                             help="Gebieden zijn genummerd op de gekozen maat, nr. 1 is het hoogst. De naam is de meest voorkomende wijk in het vak.")

    fig = go.Figure(go.Choroplethmap(
        geojson=geo, locations=vakken["vak"], z=vakken["klasse"], zmin=-0.5, zmax=n_klassen - 0.5, colorscale=schaal,
        marker_opacity=0.72, marker_line_width=0.6, marker_line_color="white", name="Vak",
        colorbar=dict(title=maat + "<br>per vak van 1 km²", tickvals=list(range(n_klassen)), ticktext=labels),
        customdata=np.stack([vakken["naam"], vakken["stations"], vakken["ritten"], vakken["docks"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} fietsstations<br>%{customdata[3]:,.0f} docks<br>"
                      "%{customdata[2]:,.0f} ritten gestart<extra></extra>"))
    if toon_fiets:
        fig.add_trace(go.Scattermap(lat=f["lat"], lon=f["lon"], mode="markers", name="Fietsstation", showlegend=True,
                                    marker=dict(size=5, color="#52514e"), text=f["station"], hovertemplate="%{text}<extra>Fietsstation</extra>"))
    if toon_metro:
        mm = metro.dropna(subset=["lat"])
        mm = mm[mm["totaal_per_jaar"] > 0]
        fig.add_trace(go.Scattermap(lat=mm["lat"], lon=mm["lon"], mode="markers", name="Metrostation (grootte = reizigers)", showlegend=True,
                                    marker=dict(size=np.sqrt(mm["totaal_per_jaar"]) / 500 + 4, color=ORANJE, opacity=0.85),
                                    text=mm["station"], customdata=mm["totaal_per_jaar"] / 1e6,
                                    hovertemplate="%{text}<br>%{customdata:.1f} mln in- en uitstappers<extra></extra>"))

    centrum, zoom = dict(lat=51.505, lon=-0.12), 11.2
    if keuze_vak != "Geen" and keuze_vak in namen:
        v = vakken[vakken["naam"] == keuze_vak].iloc[0]
        centrum, zoom = dict(lat=(v["rij"] + 0.5) * DLAT, lon=(v["kol"] + 0.5) * DLON), 13
        fig.add_trace(go.Choroplethmap(geojson=geo, locations=[v["vak"]], z=[0], showscale=False,
                                       colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                       marker_line_width=4, marker_line_color=ROOD, hoverinfo="skip", name="Gekozen vak"))
    fig.update_layout(map=dict(style="carto-positron", center=centrum, zoom=zoom), height=560, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"))
    c2.plotly_chart(fig, width="stretch")

    if keuze_vak != "Geen" and keuze_vak in namen:
        in_vak = f[f["vak"] == vakken.loc[vakken["naam"] == keuze_vak, "vak"].iloc[0]].sort_values("starts", ascending=False)
        c1.dataframe(in_vak[["station", "starts", "metro_station", "afstand_m"]].rename(columns={
            "station": "Fietsstation", "starts": "Ritten", "metro_station": "Dichtste metro", "afstand_m": "Meter"}),
            hide_index=True, width="stretch", height=200)

    st.divider()
    c1, c2 = st.columns([1, 3])
    grens = c1.slider("Afstandsgrens fietsstation tot metro (m)", 100, 600, 250, 50,
                      help="Een fietsstation telt mee voor een metrostation als het binnen deze afstand ligt.")
    m, binnen = metro_index(fiets, metro, grens)
    c1.metric("Fietsstations binnen de grens", f"{len(binnen)} van {len(fiets)}")
    metro_werkdag = metro["in_dinsdag_donderdag"].sum()
    extra_staking = stakingsdagen["ritten"].median() - normaal_werkdag
    c1.metric("Fietsritten per 100 metroreizen", nl(normaal_werkdag / metro_werkdag * 100, 1),
              help=f"Op een gewone werkdag (di tot do): {nl(metro_werkdag / 1e6, 1)} mln keer inchecken bij de metro, "
                   f"{nl(normaal_werkdag)} fietsritten.")
    groep = m.groupby(["zone_nr", "groep"], observed=True)["index"].agg(["median", "size"]).reset_index()
    fig = go.Figure()
    for g, kleur in zip(["40 fietsdocks of minder", "Meer dan 40 fietsdocks"], [GRIJS, BLAUW]):
        s = groep[groep["groep"] == g]
        fig.add_trace(go.Bar(x=["Zone " + str(int(z)) for z in s["zone_nr"]], y=s["median"], name=g, marker_color=kleur,
                             customdata=s["size"], text=[nl(v, 2) for v in s["median"]], textposition="outside",
                             hovertemplate="%{x}<br>index %{y:.2f}<br>%{customdata} metrostations<extra>" + g + "</extra>"))
    fig.add_hline(y=1, line_dash="dot", line_color=GRIJS, annotation_text="Zo druk als gemiddeld in de zone", annotation_position="top left")
    fig.update_layout(barmode="group", title=f"Metrostations met veel fietsdocks binnen {grens} m zijn drukker dan hun zone",
                      yaxis_title="Metrogebruik t.o.v. zonemediaan", bargap=0.25, bargroupgap=0.05)
    c2.plotly_chart(opmaak(fig, 400), width="stretch")
    st.markdown(f"**De fiets vult de metro aan bij de drukste stations. Bij een staking komen er ongeveer {nl(extra_staking)} "
                f"fietsritten bij, tegenover {nl(metro_werkdag / 1e6, 1)} mln metroreizen op een gewone werkdag.**")
    with st.expander("Toelichting"):
      st.markdown(
        "Index 1 = zo druk als een gemiddeld metrostation in dezelfde zone. Zo vergelijken we eerlijk: een station in het centrum "
        "is altijd drukker dan een in de buitenwijk.  \n"
        "TfL heeft de fietsen vooral bij de drukste metrostations gezet, zoals Waterloo, Holborn, "
        "Victoria en King's Cross. Op een gewone dag **vult** de fiets de metro dus aan, voor de laatste kilometer vanaf een druk station. "
        "Op dezelfde noemer gezet is de fiets klein: op een werkdag zijn er minder dan 1 fietsrit per 100 metroreizen. "
        f"Bij een staking komen er ongeveer {nl(extra_staking)} fietsritten bij, terwijl de metro normaal "
        f"{nl(metro_werkdag / 1e6, 1)} mln reizigers per dag vervoert. De fiets **springt in**, maar vervangt de metro niet.  \n"
        "Let op: dit is samenhang, geen oorzaak. De docks staan er juist omdat het daar druk is. Het patroon blijft staan "
        "als je de afstandsgrens verschuift."
    )

# Tab 4: data en methode
with tab4:
    st.subheader("Wat zit er in de data, en wat hebben we ermee gedaan?")
    bronnen = pd.DataFrame([
        ["Fietsritten 2022", "TfL, cycling.data.tfl.gov.uk (53 weekbestanden, zelf opgehaald met script)",
         f"{nl(bestanden['rijen_in_2022'].sum())} ritten", "Samengevat per dag, uur, station en route"],
        ["Fietsstations", "TfL BikePoint-API (zelf opgehaald)", f"{len(fiets)} met locatie", "Locatie en aantal docks"],
        ["Metrogebruik 2022", "Brightspace (TfL-jaartellingen)", f"{len(metro)} stations", "Jaartotaal in- en uitstappers"],
        ["Weer Londen", "Brightspace", f"{nl(len(weer_ruw))} dagen", "Alleen 2022 gebruikt"],
        ["Metrostakingen", "Wikipedia, RNOH", f"{len(STAKINGEN)} dagen", "Handmatig toegevoegd"],
    ], columns=["Dataset", "Bron", "Omvang", "Gebruik"])
    st.dataframe(bronnen, hide_index=True, width="stretch")

    ruw = df["ritten_ruw"].sum()
    kort, lang = df["weg_te_kort"].sum(), df["weg_te_lang"].sum()
    zonder_loc = alle_stations[~alle_stations["station"].isin(fiets["station"])]
    st.markdown("**Inspectie: wat we vonden, wat we deden en wat het kost aan observaties**")
    inspectie = pd.DataFrame([
        ["Fietsritten", f"Twee layouts: {(bestanden['layout'] == 'oud').sum()} bestanden oud, {(bestanden['layout'] == 'nieuw').sum()} nieuw "
         "(andere kolommen, seconden of milliseconden, andere datumnotatie)", "Omgezet naar dezelfde kolommen", "geen"],
        ["Fietsritten", "Ritten korter dan 1 minuut, alleen in het nieuwe systeem", "Verwijderd: fiets meteen teruggezet", f"-{nl(kort)}"],
        ["Fietsritten", "Ritten langer dan 24 uur", "Verwijderd: fiets niet goed teruggezet", f"-{nl(lang)}"],
        ["Fietsritten", "Dubbele rit-ID's over alle bestanden", "Gecontroleerd", "0 gevonden"],
        ["Fietsritten", "10 en 11 september: 4 en 7 ritten (systeemwissel)", "Leeg gelaten, als gat zichtbaar", "2 dagen"],
        ["Fietsritten", "Uitschieters tot 66.000 ritten per dag", "Laten staan: het zijn metrostakingen", "geen"],
        ["Fietsstations", f"{len(zonder_loc)} stations uit de ritdata niet in de BikePoint-feed", "Weggelaten op de kaart",
         f"{nl(zonder_loc['starts'].sum() / alle_stations['starts'].sum() * 100, 2)}% van de ritten"],
        ["Metro", "11 stations zonder coördinaat (zoals Bank and Monument)", "Opgezocht via de TfL StopPoint-API", "geen"],
        ["Metro", f"{(metro['totaal_per_jaar'] <= 0).sum()} rijen met 0 reizigers: stations die per vervoerstype dubbel in het bestand staan, "
         "de reizigers staan bij één rij", "Niet gebruikt bij koppelen en metro-index", f"{(metro['totaal_per_jaar'] <= 0).sum()} rijen"],
        ["Weer", "Zonuren 100%, sneeuw 98%, windstoten 82% leeg", "Kolommen niet gebruikt", "geen, 2022 compleet op 1 dag na"],
    ], columns=["Dataset", "Wat we vonden", "Wat we deden", "Effect op aantal"])
    st.dataframe(inspectie, hide_index=True, width="stretch")
    st.markdown(f"Van de **{nl(ruw)}** ritten die in 2022 begonnen, houden we er **{nl(ruw - kort - lang)}** over "
                f"(**{nl((ruw - kort - lang) / ruw * 100, 1)}%**). De conclusies hangen dus niet af van wat we weghaalden.")

    with st.expander("Uitschieters die we bewust laten staan"):
        a_, _, _ = model_fit(df, True)
        b_, _, _ = model_fit(df, False)
        fa = a_[a_["set"].str.startswith("Toets")]["fout"].abs().mean()
        fb = b_[b_["set"].str.startswith("Toets")]["fout"].abs().mean()
        st.markdown(
            "De drukste dagen van het jaar lijken fouten, maar het zijn metrostakingen. Die laten we staan: ze zijn echt en ze zijn "
            "een van onze bevindingen. Bij de weertrend laten we ze weg, bij de voorspelling kun je kiezen. "
            f"De gemiddelde fout op november en december is {nl(fa)} zonder en {nl(fb)} met stakingen in de training: "
            "de conclusie hangt dus niet van die keuze af."
        )

    with st.expander("Koppelen van fiets en metro"):
        zonder_loc = alle_stations[~alle_stations["station"].isin(fiets["station"])]
        st.markdown(
            "Namen van fiets- en metrostations matchen niet, dus koppelen we op afstand: per fietsstation het dichtstbijzijnde "
            "metrostation (haversine-afstand in meters).\n\n"
            f"- **{len(zonder_loc)} fietsstations** uit de ritdata staan niet meer in de BikePoint-feed (opgeheven, hernoemd of werkplaats). "
            f"Samen {nl(zonder_loc['starts'].sum())} ritten, {nl(zonder_loc['starts'].sum() / alle_stations['starts'].sum() * 100, 2)}% van het totaal.\n"
            "- **11 metrostations** hadden geen coördinaat (dubbele stations zoals Bank and Monument). Die hebben we opgezocht via de "
            "TfL StopPoint-API in plaats van ze weg te gooien, want het zijn grote knooppunten.\n"
            f"- {(metro['totaal_per_jaar'] <= 0).sum()} rijen hebben 0 reizigers: stations die per vervoerstype dubbel in het bestand staan "
            "(zoals Stratford en Liverpool Street). Die rijen slaan we over bij het koppelen, anders zou een fietsstation aan een lege rij hangen.\n"
            "- Voor de metro-index gebruiken we alleen zone 1 en 2, waar de fietsen staan."
        )
        st.dataframe(metro[metro["coord_bron"] != "bestand"][["station", "vervoerstype", "lat", "lon", "coord_bron"]],
                     hide_index=True, width="stretch")

    with st.expander("Weer en voorspelmodel"):
        miss = weer_ruw.drop(columns=weer_ruw.columns[0]).isna().mean().mul(100).round(1)
        st.dataframe(miss.rename("% leeg (2000 t/m 2023)").to_frame().T, width="stretch")
        st.markdown("Zonuren (100% leeg), sneeuw (98%) en windstoten (82%) laten we vallen. Voor 2022 zijn temperatuur, neerslag en "
                    "wind compleet op één dag na. Neerslag nemen we in het model als log(1 + mm), zodat één extreme regendag "
                    "(27,5 mm op 3 november) het model niet overheerst. Daglengte rekenen we zelf uit de datum uit. "
                    "Na de systeemwissel liggen de aantallen ongeveer 5.000 ritten per dag lager dan het weer verklaart; of dat "
                    "registratie is of echt minder fietsen, kunnen we niet zien. Daarom zit het als aparte variabele in het model.")
        res_t, coef_t, _ = model_fit(df, True)
        st.dataframe(coef_t.round(0).astype(int).rename("Effect op ritten per dag").to_frame(), width="stretch")

    with st.expander("Wat we bewust hebben weggelaten"):
        r_ = routes[~routes["rondrit"]].head(3)
        st.markdown(
            "- **Drukste routes.** De top bestaat uit ritjes in Hyde Park en het Olympic Park, bijvoorbeeld "
            f"{r_.iloc[0]['startstation'].split(',')[0]} naar {r_.iloc[0]['eindstation'].split(',')[0]} ({nl(r_.iloc[0]['ritten'])} ritten). "
            "Dat is recreatie en zegt niets over de metro.\n"
            "- **Metrogebruik per dagsoort.** We gebruiken het jaartotaal; per dagsoort voegt weinig toe en maakt de kaart onrustig.\n"
            "- **Kaggle-metrodata 2007 tot 2017.** Een andere periode dan de fietsdata, dus niet te koppelen.\n"
            "- **Alle stations als losse punten.** Op de kaart tonen we vakken; losse stations kun je aanzetten.\n\n"
            "Zo blijft elk tabblad bij één vraag."
        )

    with st.expander("Code en bronnen"):
        st.markdown(
            "- Data ophalen: `01_fietsdata_ophalen.py` (ritten) en `02_stations_koppelen.py` (stationslocaties en koppeling). "
            "Die draaien eenmalig lokaal; het dashboard gebruikt alleen de samenvattingen in de map data.\n"
            "- Haversine-formule: [Wikipedia, Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula), omgezet naar numpy.\n"
            "- Daglengte: [Wikipedia, Sunrise equation](https://en.wikipedia.org/wiki/Sunrise_equation), vereenvoudigd.\n"
            "- Kaart: [Plotly, Choroplethmap](https://plotly.com/python/tile-county-choropleth/), aangepast naar eigen vakken van 1 km.\n"
            "- Klikken in grafieken: [Streamlit, st.plotly_chart met on_select](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart).\n"
            "- Stakingsdagen: [Wikipedia, London Underground strikes](https://en.wikipedia.org/wiki/London_Underground_strikes) en "
            "[RNOH, tube strike 1 en 3 maart](https://rnoh.nhs.uk/news/proposed-tube-strike-tuesday-1-march-and-thursday-3-march)."
        )
