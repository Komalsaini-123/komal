#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import math
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ------------------------------
# Helpers
# ------------------------------
def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    # keep letters, numbers, commas and spaces
    s = re.sub(r"[^a-z0-9,+&/()' -]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def coalesce_cols(df, colnames):
    for c in colnames:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_dataframe(path_or_buffer):
    if isinstance(path_or_buffer, str):
        df = pd.read_csv(path_or_buffer)
    else:
        df = pd.read_csv(path_or_buffer)
    # standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def build_corpus(df):
    # pick best-available columns for textual features
    name_col = coalesce_cols(df, ["name", "dishname", "title", "recipe_name"])
    ing_col  = coalesce_cols(df, ["ingredients", "ingredient", "ingr", "recipe_ingredients"])
    course_col = coalesce_cols(df, ["course", "meal_type", "category"])
    diet_col = coalesce_cols(df, ["diet", "veg_non_veg", "veg_nonveg"])
    flavor_col = coalesce_cols(df, ["flavor_profile", "flavor", "taste"])
    state_col = coalesce_cols(df, ["state", "origin_state"])
    region_col = coalesce_cols(df, ["region"])
    instructions_col = coalesce_cols(df, ["instructions", "steps", "method", "directions"])

    usable_cols = [c for c in [name_col, ing_col, course_col, diet_col, flavor_col, state_col, region_col, instructions_col] if c]

    if not usable_cols:
        raise ValueError("No textual columns found. Expected any of: name, ingredients, course, diet, flavor_profile, state, region, instructions.")

    # create a combined text field
    def row_text(row):
        parts = []
        if name_col: parts.append(str(row[name_col]))
        if ing_col: parts.append(str(row[ing_col]))
        if course_col: parts.append(str(row[course_col]))
        if diet_col: parts.append(str(row[diet_col]))
        if flavor_col: parts.append(str(row[flavor_col]))
        if state_col: parts.append(str(row[state_col]))
        if region_col: parts.append(str(row[region_col]))
        if instructions_col: parts.append(str(row[instructions_col])[:400])  # cap to reduce noise
        return normalize_text(" , ".join(parts))

    df = df.copy()
    df["__text__"] = df.apply(row_text, axis=1)

    # basic filters list
    filters = {}
    if course_col: filters["course"] = course_col
    if diet_col: filters["diet"] = diet_col
    if flavor_col: filters["flavor_profile"] = flavor_col
    if state_col: filters["state"] = state_col
    if region_col: filters["region"] = region_col

    # best title column
    title_col = name_col if name_col else usable_cols[0]
    return df, filters, title_col, ing_col

@st.cache_resource(show_spinner=False)
def fit_vectorizer(corpus_texts):
    # word n-grams for ingredient matching
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95
    )
    X = vectorizer.fit_transform(corpus_texts)
    return vectorizer, X

def search_similar(query, vectorizer, X, df, title_col, top_k=20, mask=None):
    q = normalize_text(query)
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, X).ravel()

    idx = np.arange(len(df))
    if mask is not None:
        idx = idx[mask]

    # argsort descending
    top_idx = np.argsort(-sims[idx])[:top_k]
    picked = idx[top_idx]
    out = df.iloc[picked].copy()
    out["similarity"] = sims[picked]
    # reorder columns
    cols = [c for c in df.columns if c not in ["__text__", "similarity"]]
    out = out[[*cols, "similarity"]]
    return out

def apply_filters(df, filters, selections):
    if not selections:
        return np.ones(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    for nice_name, col in filters.items():
        chosen = selections.get(nice_name)
        if chosen:
            mask &= df[col].astype(str).str.lower().isin([str(x).lower() for x in chosen])
    return mask

def boost_by_rating(df, base_scores, rating_cols=("rating","ratings","avg_rating","average_rating","stars")):
    # If a known rating column exists, combine with similarity using a soft boost
    found = None
    for c in df.columns:
        if c.lower() in rating_cols:
            found = c; break
    if not found:
        return base_scores
    r = pd.to_numeric(df[found], errors="coerce")
    r = (r - np.nanmin(r)) / (np.nanmax(r) - np.nanmin(r) + 1e-9)  # 0..1
    return 0.85*base_scores + 0.15*r

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Indian Food Recommender", page_icon="🍲", layout="wide")

st.title("🍲 Indian Food Recommendation System")
st.caption("Content-based recommendations using TF-IDF + cosine similarity over ingredients and metadata.")

# Data input
st.sidebar.header("1) Load dataset")
uploaded = st.sidebar.file_uploader("Upload CSV (Indian food dataset)", type=["csv"])
default_path = st.sidebar.text_input("...or path to CSV (server file system)", value="C:\\Users\\Komal\\Downloads\\indian_food.csv")
use_uploaded = uploaded is not None
go = st.sidebar.button("Load / Reload")

if 'df' not in st.session_state or go:
    try:
        if use_uploaded:
            df = load_dataframe(uploaded)
        else:
            if not os.path.exists(default_path):
                st.info("No file at 'data/indian_food.csv'. Upload a CSV or set a valid path in the sidebar.")
                st.stop()
            df = load_dataframe(default_path)

        df, filters, title_col, ing_col = build_corpus(df)
        vectorizer, X = fit_vectorizer(df["__text__"].tolist())

        st.session_state.df = df
        st.session_state.filters = filters
        st.session_state.title_col = title_col
        st.session_state.ing_col = ing_col
        st.session_state.vectorizer = vectorizer
        st.session_state.X = X
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

df = st.session_state.df
filters = st.session_state.filters
title_col = st.session_state.title_col
ing_col = st.session_state.ing_col
vectorizer = st.session_state.vectorizer
X = st.session_state.X

# Sidebar filters
st.sidebar.header("2) Filters")
selections = {}
for nice_name, col in filters.items():
    options = sorted([o for o in df[col].dropna().astype(str).unique() if o != ""], key=lambda x: x.lower())
    if options:
        chosen = st.sidebar.multiselect(f"{nice_name.capitalize()}", options=options, default=[])
        selections[nice_name] = chosen

# Query Controls
st.sidebar.header("3) Query")
mode = st.sidebar.radio("Search by:", ["Dish name", "Ingredients / free text"], index=0)

if mode == "Dish name":
    options = df[title_col].dropna().astype(str).unique().tolist()
    options = sorted(options, key=lambda x: x.lower())[:5000]  # cap for performance
    dish = st.selectbox("Select a dish", options)
    query = dish
else:
    query = st.text_input("Enter ingredients or what you're craving", value="paneer, tomato, onion, spices")

top_k = st.sidebar.slider("Number of recommendations", min_value=5, max_value=30, value=10, step=1)

# Recommend
if st.button("Recommend"):
    mask = apply_filters(df, filters, selections)
    results = search_similar(query, vectorizer, X, df, title_col, top_k=top_k*3, mask=mask)

    # (Optional) boost by ratings if present
    if "similarity" in results.columns:
        boosted = boost_by_rating(results, results["similarity"].values)
        results = results.assign(score=boosted).sort_values("score", ascending=False).head(top_k)
    else:
        results = results.head(top_k)

    # Show
    st.success(f"Top {len(results)} recommendations for: **{query}**")
    show_cols = [c for c in results.columns if c not in ["__text__", "score"]]
    st.dataframe(results[show_cols].reset_index(drop=True), use_container_width=True)

    # Explainability: show top keywords for query match (simple)
    st.subheader("Why these? (top terms)")
    try:
        # show largest TF-IDF terms in the query vector
        q_vec = vectorizer.transform([normalize_text(query)])
        feature_names = np.array(vectorizer.get_feature_names_out())
        arr = q_vec.toarray()[0]
        top_idx = np.argsort(arr)[-15:][::-1]
        terms = [feature_names[i] for i in top_idx if arr[i] > 0]
        st.write(", ".join(terms) if terms else "Terms not available.")
    except Exception:
        st.write("Terms not available.")

# Peek data
with st.expander("Preview dataset & detected columns"):
    st.write(f"Detected title column: `{title_col}`  |  ingredients column: `{ing_col}`")
    st.dataframe(df.head(20), use_container_width=True)

st.sidebar.caption("Tip: Use filters + ingredient search for best results. If ratings exist, they will be used to slightly boost the ranking.")



# In[ ]:





# In[ ]:




