import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Letterboxd recommender demo", layout="wide")

DATA = Path(__file__).parent
if (DATA / "app_data").exists():
    DATA = DATA / "app_data"


@st.cache_data
def load_json(name):
    with open(DATA / name) as f:
        return json.load(f)


@st.cache_data
def load_films():
    films = pd.read_csv(DATA / "films.csv")
    return films.set_index("film_id")


history = load_json("history.json")
films = load_films()


def title(film_id):
    """Readable title: 'Name (year)'"""
    if film_id not in films.index:
        return film_id
    row = films.loc[film_id]
    name = row["film_name"] if pd.notna(row["film_name"]) else film_id
    year = f" ({int(row['year'])})" if pd.notna(row["year"]) else ""
    return f"{name}{year}"


st.title("Letterboxd movie recommender demo")
st.caption("Users are pseudonymized. Recommendations are precomputed.")

user = st.selectbox("Choose a user", list(history.keys()))

st.subheader("Films this user liked")
for film_id in history[user]:
    st.write("•", title(film_id))
