import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Qualité Radio LTE", page_icon="📡", layout="wide")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "logistic_regression.pkl"
SCALER_PATH = BASE / "models" / "scaler.pkl"
DATASET_PATH = BASE / "dataset" / "dataset_radio.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)

st.title("📡 Prédiction et cartographie de la qualité radio")
st.markdown("### Projet Master — Machine Learning appliqué aux réseaux mobiles")

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not DATASET_PATH.exists():
    st.error("Fichiers du projet manquants. Vérifiez les dossiers models/ et dataset/.")
    st.stop()

logistic, scaler = load_model()
df = load_data()

tab1, tab2 = st.tabs(["🎯 Prédiction", "🗺️ Carte radio"])

with tab1:
    st.subheader("Mesures radio")
    c1, c2, c3 = st.columns(3)

    with c1:
        rsrp = st.number_input("RSRP (dBm)", value=-80.0, step=1.0)
        rsrq = st.number_input("RSRQ (dB)", value=-9.0, step=1.0)

    with c2:
        sinr = st.number_input("SINR (dB)", value=20.0, step=1.0)
        download = st.number_input("Download (Mbps)", value=100.0, min_value=0.0, step=1.0)

    with c3:
        upload = st.number_input("Upload (Mbps)", value=20.0, min_value=0.0, step=1.0)
        latency = st.number_input("Latence (ms)", value=40.0, min_value=0.0, step=1.0)

    if st.button("🔎 PRÉDIRE LA QUALITÉ", use_container_width=True):
        X_new = pd.DataFrame({
            "rsrp": [rsrp], "rsrq": [rsrq], "sinr": [sinr],
            "download_mbps": [download], "upload_mbps": [upload],
            "latency_ms": [latency]
        })
        prediction = logistic.predict(scaler.transform(X_new))[0]

        if prediction == "Bonne":
            st.success("🟢 QUALITÉ RADIO : BONNE")
        elif prediction == "Moyenne":
            st.warning("🟡 QUALITÉ RADIO : MOYENNE")
        else:
            st.error("🔴 QUALITÉ RADIO : MAUVAISE")

with tab2:
    st.subheader("🗺️ Carte de qualité radio simulée")
    st.caption("Les coordonnées sont synthétiques et servent uniquement à la démonstration.")

    # Filtres
    classes = st.multiselect(
        "Afficher les classes",
        options=["Bonne", "Moyenne", "Mauvaise"],
        default=["Bonne", "Moyenne", "Mauvaise"]
    )
    map_df = df[df["quality"].isin(classes)].copy()

    # Centre approximatif de la zone simulée
    center = [map_df["latitude"].mean(), map_df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=12, control_scale=True)

    colors = {"Bonne": "green", "Moyenne": "orange", "Mauvaise": "red"}

    # Tous les points (5000 max : acceptable pour cette démonstration)
    for _, row in map_df.iterrows():
        q = row["quality"]
        popup = folium.Popup(
            f"""
            <b>Qualité :</b> {q}<br>
            <b>RSRP :</b> {row['rsrp']:.2f} dBm<br>
            <b>RSRQ :</b> {row['rsrq']:.2f} dB<br>
            <b>SINR :</b> {row['sinr']:.2f} dB<br>
            <b>Download :</b> {row['download_mbps']:.2f} Mbps<br>
            <b>Upload :</b> {row['upload_mbps']:.2f} Mbps<br>
            <b>Latence :</b> {row['latency_ms']:.2f} ms
            """,
            max_width=300
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=colors[q],
            fill=True,
            fill_color=colors[q],
            fill_opacity=0.65,
            popup=popup
        ).add_to(m)

    st_folium(m, width=None, height=650)

    st.markdown("**Légende :** 🟢 Bonne &nbsp;&nbsp; 🟠 Moyenne &nbsp;&nbsp; 🔴 Mauvaise")
