import streamlit as st
import pandas as pd
from datetime import datetime, date

# ========== PAGE SETUP ==========
st.set_page_config(page_title="Sentiment Analysis Maybank Marathon 2024", layout="wide")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("data/sentiment_data.csv", parse_dates=["Tanggal"])
    return df

df = load_data()

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_base64 = get_base64_image("maybank_logo.png")


# ========== HEADER ==========
col1, col2 = st.columns([1, 4])

with col1:
    st.markdown(
        f"""
        <style>
        .logo-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100px;
            overflow: hidden;
        }}
        .logo-container img {{
            height: auto;
            width: auto;
            max-height: 100px;
            object-fit: contain;
        }}
        </style>
        <div class="logo-container">
            <img src="data:image/png;base64,{image_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="display: flex; align-items: center; height: 100px;">
            <h1 style='margin: 0; padding: 0;'>Sentiment Analysis Review: Maybank Marathon 2024</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ========== TOP INFO SECTION ==========
left_col, center_col, right_col = st.columns([3, 3, 2])

with left_col:
    st.markdown("### 🟨 Latar Belakang")
    st.markdown("Deskripsi singkat tentang pentingnya analisis sentimen untuk acara Maybank Marathon 2024.")

with center_col:
    st.markdown("### 🟨 Tujuan")
    st.markdown("Tujuan dari project ini adalah memahami bagaimana masyarakat merespons Maybank Marathon.")

with right_col:
    st.markdown("### Filters")

    # Filter Tanggal
    min_date = df["Tanggal"].min().date()
    max_date = df["Tanggal"].max().date()
    selected_date = st.date_input("Tanggal", (min_date, max_date), min_value=min_date, max_value=max_date)

    # Filter Platform
    selected_platform = st.selectbox("Platform", ["All"] + sorted(df["Platform"].unique().tolist()))

# ========== FILTERING DATA ==========

filtered_df = df.copy()

# Handle tanggal filter
try:
    if isinstance(selected_date, tuple):
        start_date, end_date = selected_date
    else:
        # Jika user hanya pilih 1 tanggal
        start_date = end_date = selected_date

    # Validasi rentang tanggal
    if start_date < min_date or end_date > max_date:
        st.error("⚠️ Please select the date range between **June to October 2024**.")
        st.stop()
    
    # Filter by date
    filtered_df = filtered_df[
        (filtered_df["Tanggal"] >= pd.to_datetime(start_date)) &
        (filtered_df["Tanggal"] <= pd.to_datetime(end_date))
    ]

except Exception as e:
    st.error("⚠️ Please select a valid date range between **June to October 2024**.")
    st.stop()


# Filter by platform
if selected_platform != "All":
    filtered_df = filtered_df[filtered_df["Platform"] == selected_platform]


import streamlit.components.v1 as components
from PIL import Image

# ========== METRICS CARD ==========

# Hitung jumlah total dan per sentimen
total_comments = len(filtered_df)
positif_count = (filtered_df["Sentiment"] == "positif").sum()
netral_count = (filtered_df["Sentiment"] == "netral").sum()
negatif_count = (filtered_df["Sentiment"] == "negatif").sum()

# Hitung tren (dibandingkan hari sebelumnya jika memungkinkan)

# METRICS 
filtered_df_sorted = filtered_df.sort_values("Tanggal")
if len(filtered_df_sorted["Tanggal"].unique()) >= 2:
    yesterday = filtered_df_sorted["Tanggal"].max()
    prev_day = filtered_df_sorted["Tanggal"].unique()[-2]

    df_today = filtered_df[filtered_df["Tanggal"] == yesterday]
    df_prev = filtered_df[filtered_df["Tanggal"] == prev_day]

    def calc_trend(current, previous):
        if previous == 0:
            return "⬆️" if current > 0 else "-"
        return "⬆️" if current > previous else ("⬇️" if current < previous else "➡️")

    total_trend = calc_trend(len(df_today), len(df_prev))
    pos_trend = calc_trend((df_today["Sentiment"] == "positif").sum(), (df_prev["Sentiment"] == "positif").sum())
    net_trend = calc_trend((df_today["Sentiment"] == "netral").sum(), (df_prev["Sentiment"] == "netral").sum())
    neg_trend = calc_trend((df_today["Sentiment"] == "negatif").sum(), (df_prev["Sentiment"] == "negatif").sum())
else:
    total_trend = pos_trend = net_trend = neg_trend = "-"

# Tampilkan dengan layout 4 kolom
card1, card2, card3, card4 = st.columns(4)

with card1:
    st.markdown(f"#### Total Komentar")
    st.metric(label="Total", value=total_comments, delta=total_trend)

with card2:
    st.markdown(f"#### Sentimen Positif")
    st.metric(label="Jumlah", value=positif_count, delta=pos_trend)

with card3:
    st.markdown(f"#### Sentimen Netral")
    st.metric(label="Jumlah", value=netral_count, delta=net_trend)

with card4:
    st.markdown(f"#### Sentimen Negatif")
    st.metric(label="Jumlah", value=negatif_count, delta=neg_trend)


# Styling per card berdasarkan urutan
st.markdown("""
<style>
/* Semua kartu */
[data-testid="metric-container"] {
    border-radius: 12px;
    padding: 15px;
    margin: 5px;
    color: black;
    text-align: center;
}

/* Card 1 - Total (abu terang) */
div[data-testid="metric-container"]:nth-child(1) {
    background-color: #f8f9fa;
}
/* Card 2 - Positif (hijau lembut) */
div[data-testid="metric-container"]:nth-child(2) {
    background-color: #d1fae5;
}
/* Card 3 - Netral (abu keunguan lembut) */
div[data-testid="metric-container"]:nth-child(3) {
    background-color: #e5e7eb;
}
/* Card 4 - Negatif (merah lembut) */
div[data-testid="metric-container"]:nth-child(4) {
    background-color: #fee2e2;
}
</style>
""", unsafe_allow_html=True)


import altair as alt


st.markdown("##  Tren Komentar")

# Tambahkan keterangan platform yang sedang difilter
if selected_platform != "All":
    st.markdown(f"<p style='color: gray;'>Menampilkan tren komentar hanya untuk platform: <b>{selected_platform}</b></p>", unsafe_allow_html=True)
else:
    st.markdown(f"<p style='color: gray;'>Menampilkan tren komentar dari semua platform</p>", unsafe_allow_html=True)


# --- Preprocessing untuk visualisasi ---
trend_df = (
    filtered_df.groupby("Tanggal")
    .agg(
        total_comments=("standardized_text", "count"),
        negative_comments=("Sentiment", lambda x: (x == "negatif").sum())
    )
    .reset_index()
)

# Tambahkan kolom rasio komentar negatif dalam persen
trend_df["Negativity Ratio(%)"] = (trend_df["negative_comments"] / trend_df["total_comments"]) * 100

# --- Chart: Bar (Jumlah Komentar) + Line (Negativity Ratio) ---
base = alt.Chart(trend_df).encode(
    x=alt.X("Tanggal:T", axis=alt.Axis(title="Tanggal", format="%d %b"))
)

# Bar chart: Jumlah komentar
bar = base.mark_bar(color="#facc15").encode(
    y=alt.Y("total_comments:Q", axis=alt.Axis(title="Jumlah Komentar"))
)

# Line chart: Negativity ratio
line = base.mark_line(color="#e11d48", strokeWidth=3).encode(
    y=alt.Y("Negativity Ratio(%):Q", axis=alt.Axis(title="Negativity Ratio (%)"))
)

# Point chart: Titik pada garis
points = base.mark_circle(color="#e11d48", size=60).encode(
    y=alt.Y("Negativity Ratio(%):Q")
)

# Gabungkan dan atur sumbu agar tidak tumpang tindih
combo_chart = alt.layer(bar, line, points).resolve_scale(
    y="independent"  # Pakai sumbu y terpisah untuk bar dan line
)

st.altair_chart(combo_chart, use_container_width=True)


st.markdown("## Sentiment Analysis")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# ========== POPULAR WORDS ANALYSIS SECTION ==========
st.markdown("##  Popular Words Analysis")

# Kolom kiri-kanan untuk filter
col1, col2 = st.columns([1, 1])

with col1:
    sentiment_option = st.selectbox(" Pilih Sentimen:", ["All", "positif", "netral", "negatif"], index=0)

with col2:
    ngram_option = st.radio("🔗 Pilih N-gram:", ["Bigram", "Trigram", "Fourgram"], horizontal=True)

# Bersihkan emoji dari teks
def remove_emotes(text):
    return re.sub(r'\[.*?\]', '', text) if isinstance(text, str) else text

df['text_clean'] = df['standardized_text'].apply(remove_emotes)

# Filter sentimen
filtered_df = df.copy()
if sentiment_option != "All":
    filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment_option]

# Stopwords
stopwords_list = stopwords.words('indonesian')

# Sentimen dominan per n-gram
def get_dominant_sentiment_ngram(ngram, df):
    hits = df[df['text_clean'].str.contains(re.escape(ngram), na=False)]
    if hits.empty:
        return 'netral'
    return hits['Sentiment'].value_counts().idxmax()

# Plot pakai plotly
def plot_ngram_chart_plotly(df, text_col, n=2):
    corpus = df[text_col].dropna().tolist()
    vec = CountVectorizer(ngram_range=(n, n), stop_words=stopwords_list, min_df=2)
    X = vec.fit_transform(corpus)
    ngram_freq = X.sum(axis=0).A1
    ngram_names = vec.get_feature_names_out()

    ngram_df = pd.DataFrame({'ngram': ngram_names, 'count': ngram_freq})
    ngram_df = ngram_df.sort_values('count', ascending=False).head(20)

    # Tambahkan kolom sentimen dominan dan warna
    ngram_df['sentiment'] = ngram_df['ngram'].apply(lambda x: get_dominant_sentiment_ngram(x, df))
    color_map = {'positif': '#2ecc71', 'netral': '#3498db', 'negatif': '#e74c3c'}
    ngram_df['color'] = ngram_df['sentiment'].map(color_map)

    # Buat plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ngram_df['ngram'][::-1],
        x=ngram_df['count'][::-1],
        orientation='h',
        marker_color=ngram_df['color'][::-1],
        hovertext=ngram_df['sentiment'][::-1],
        hoverinfo='text+x+y'
    ))

    fig.update_layout(
        title=f"Top {n}-grams berdasarkan Frekuensi & Sentimen Dominan",
        xaxis_title="Frekuensi",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor='white',
        height=600,
        margin=dict(l=100, r=30, t=50, b=30),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Jalankan chart
ngram_value = {"Bigram": 2, "Trigram": 3, "Fourgram": 4}[ngram_option]
plot_ngram_chart_plotly(filtered_df, 'text_clean', n=ngram_value)


# ========== TOPIC MODELLING SECTION ==========
st.markdown("## 🔍 Topic Modelling Analysis")

# Tambahkan CSS untuk styling
st.markdown("""
<style>
/* Styling untuk tab */
[role="tab"] {
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
}

/* Styling untuk visualisasi */
[data-testid="stHorizontalBlock"] {
    margin-bottom: 20px;
}

/* Judul section */
h2 {
    border-bottom: 2px solid #f0f2f6;
    padding-bottom: 8px;
    margin-top: 30px !important;
}
</style>
""", unsafe_allow_html=True)