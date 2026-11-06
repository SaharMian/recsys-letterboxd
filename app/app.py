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


RESULTS = Path(__file__).parent.parent / "results"


@st.cache_data
def load_table(name):
    return pd.read_csv(RESULTS / name, index_col=0)


st.title("Letterboxd movie recommender demo")
st.caption("Users are pseudonymized. Recommendations are precomputed by the models "
           "in the notebooks. Films the user has already rated are never recommended.")

tab_recs, tab_results, tab_about = st.tabs(["Recommendations", "Results", "About"])

# ---------------------------------------------------------------- Recommendations
with tab_recs:
    user = st.selectbox("Choose a user", list(history.keys()))

    st.subheader("Films this user liked")
    cols = st.columns(6)
    for k, film_id in enumerate(history[user]):
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

# ---------------------------------------------------------------- Results
with tab_results:
    st.subheader("Main task (test set)")
    st.caption("For each user the model ranks all films not yet seen, and we check how many "
               "of the films the user liked (rating 4.0 or higher) in the held-out test ratings "
               "appear in the top 10. NDCG@10 rewards hits near the top. Coverage is the share "
               "of all films that are recommended to anyone.")
    try:
        test = load_table("results_test.csv")
        st.dataframe(test.round(4))
        st.bar_chart(test["NDCG@10"], horizontal=True)
    except FileNotFoundError:
        st.warning("results_test.csv not found")

    st.markdown(
        "- **NCF is the best model**, about 32% better than the popularity baseline.\n"
        "- **SVD** trained on star ratings recommends obscure films and scores close to zero. "
        "With a minimum-ratings filter it only reaches the level of popularity.\n"
        "- The **two-tower** model, which also uses film text, is slightly below NCF on this task."
    )

    st.subheader("Cold start: films with no ratings")
    st.caption("3,000 films had all their ratings removed from training. The models must rank "
               "only these new films. Random is the floor.")
    try:
        cold = load_table("results_cold_test.csv")
        cold["Users"] = cold["Users"].astype(int)
        st.dataframe(cold)
        st.bar_chart(cold["NDCG@10"], horizontal=True)
    except FileNotFoundError:
        st.warning("results_cold_test.csv not found")

    st.markdown(
        "- The two-tower model is about 12 times better than random for new films, "
        "because it can use the film text.\n"
        "- Without the text (ID only) it is no better than random.\n"
        "- A list that is the same for every user scores almost as well, so the gain comes "
        "from general taste, not from personalization."
    )

# ---------------------------------------------------------------- About
with tab_about:
    st.markdown(
        """
**Data:** Letterboxd film ratings (Kaggle), a sample of 3,000 users and 61,555 films
(4.4 million ratings). Film genres come from IMDb. Each user's ratings are split 80/10/10
into train, validation and test. A film is "liked" if rated 4.0 or higher.

**Models:** popularity baseline, SVD (matrix factorization), NCF (NeuMF) and a two-tower
model with film text embeddings.

**Limitations:** the data has no timestamps, so the split is random per user. About 15% of
the films have no genres. The two-tower results vary somewhat between training runs.

Code and full report: [GitHub](https://github.com/SaharMian/recsys-letterboxd)
"""
    )
