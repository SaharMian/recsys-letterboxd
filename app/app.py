import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Letterboxd recommender demo", layout="wide")

# The data files are either next to this file or in an app_data folder
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
    return films.set_index("film_id").to_dict("index")


history = load_json("history.json")
films = load_films()


def info(film_id):
    """Returns (title with year, genres, poster url) for a film"""
    row = films.get(film_id)
    if row is None:
        return film_id, "", None
    name = row["film_name"] if pd.notna(row["film_name"]) else film_id
    year = f" ({int(row['year'])})" if pd.notna(row["year"]) else ""
    genres = row["genres"] if pd.notna(row["genres"]) else ""
    poster = row["poster_url"] if pd.notna(row["poster_url"]) else None
    return f"{name}{year}", genres, poster


def show_film(film_id):
    """Small poster on the left, title and genres on the right"""
    title, genres, poster = info(film_id)
    c1, c2 = st.columns([1, 3])
    with c1:
        if poster:
            st.image(poster, width=70)
    with c2:
        st.markdown(f"**{title}**")
        if genres:
            st.caption(genres.replace(",", ", "))


MODELS = [
    ("Popularity", "recs_popularity.json",
     "The same top list for everyone (minus films the user has already seen)."),
    ("NCF", "recs_ncf.json",
     "Neural collaborative filtering. Learns from user and film ids only."),
    ("Two-tower", "recs_two_tower.json",
     "Also uses the film's text (title, year, genres)."),
]

st.title("Letterboxd movie recommender demo")
st.caption("Users are pseudonymized. Recommendations are precomputed by the models "
           "in the notebooks. Films the user has already rated are never recommended.")

user = st.selectbox("Choose a user", list(history.keys()))

st.subheader("Films this user liked")
liked = history[user]
cols = st.columns(6)
for k, film_id in enumerate(liked):
    title, _, poster = info(film_id)
    with cols[k % 6]:
        if poster:
            st.image(poster, width=110)
        st.caption(title)

st.subheader("Recommendations")
cols = st.columns(3)
for col, (name, file, description) in zip(cols, MODELS):
    with col:
        st.markdown(f"### {name}")
        st.caption(description)
        try:
            recs = load_json(file)[user]
        except FileNotFoundError:
            st.warning(f"{file} not found")
            continue
        for film_id in recs:
            show_film(film_id)
