import streamlit as st
import pandas as pd
from datetime import datetime, date
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# Download stopwords if not already downloaded
nltk.download('stopwords')

# ========== PAGE SETUP ==========
st.set_page_config(page_title="Sentiment Analysis Maybank Marathon 2024", layout="wide")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("data/sentiment_data.csv", parse_dates=["Tanggal"])
    return df

df = load_data()

# ========== HEADER ==========
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_base64 = get_base64_image("maybank_logo.png")

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
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("### 🟨 Deskripsi")
    st.markdown(
        """
        <div style="font-size:20px; text-align: justify;">
        Maybank Marathon merupakan salah satu ajang lomba lari berskala internasional yang diselenggarakan secara rutin di Bali sejak tahun 2012. Dengan latar pemandangan tropis Bali, event ini tidak hanya menarik minat pelari profesional maupun amatir, tetapi juga menjadi daya tarik wisata olahraga (sport tourism) yang diakui secara global. Maybank Marathon tahun 2024 berlangsung pada tanggal 25 Agustus 2024.
        </div>
        """, unsafe_allow_html=True
    )

# with center_col:
#     st.markdown("### 🟨 Tujuan")
#     st.markdown(
#         """
#         <div style="font-size:20px; text-align: justify;">
#         Menyajikan analisa sentimen publik terhadap Maybank Marathon 2024. Melalui visualisasi data dan insight yang dihasilkan, diharapkan dapat mendukung proses pengambilan keputusan berbasis data untuk meningkatkan kualitas event di tahun-tahun berikutnya.
#         </div>
#         """, unsafe_allow_html=True
#     )

with right_col:
    st.markdown("### Filters")

    # Filter Tanggal
    min_date = df["Tanggal"].min().date()
    max_date = df["Tanggal"].max().date()
    selected_date = st.date_input("Tanggal", (min_date, max_date), min_value=min_date, max_value=max_date)

    # Filter Platform
    selected_platform = st.selectbox("Platform", ["All"] + sorted(df["Platform"].unique().tolist()))

# ========== FILTERING DATA ==========
def filter_data(df, selected_date, selected_platform):
    filtered_df = df.copy()
    
    # Handle tanggal filter
    try:
        if isinstance(selected_date, tuple):
            start_date, end_date = selected_date
        else:
            start_date = end_date = selected_date

        # Filter by date
        filtered_df = filtered_df[
            (filtered_df["Tanggal"] >= pd.to_datetime(start_date)) &
            (filtered_df["Tanggal"] <= pd.to_datetime(end_date))
        ]
    except Exception as e:
        st.error("⚠️ Please select a valid date range.")
        return pd.DataFrame()

    # Filter by platform
    if selected_platform != "All":
        filtered_df = filtered_df[filtered_df["Platform"] == selected_platform]
    
    return filtered_df

filtered_df = filter_data(df, selected_date, selected_platform)

# ========== METRICS SECTION ==========
if not filtered_df.empty:
    # Hitung jumlah total dan per sentimen
    total_comments = len(filtered_df)
    positif_count = (filtered_df["Sentiment"] == "positif").sum()
    netral_count = (filtered_df["Sentiment"] == "netral").sum()
    negatif_count = (filtered_df["Sentiment"] == "negatif").sum()

    # Hitung tren (dibandingkan hari sebelumnya jika memungkinkan)
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

    # Styling per card
    # Styling per card dengan efek hover
    st.markdown("""
    <style>
    [data-testid="metric-container"] {
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
        color: black;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    div[data-testid="metric-container"]:nth-child(1) {
        background-color: #f8f9fa;
    }
    div[data-testid="metric-container"]:nth-child(2) {
        background-color: #d1fae5;
    }
    div[data-testid="metric-container"]:nth-child(3) {
        background-color: #e5e7eb;
    }
    div[data-testid="metric-container"]:nth-child(4) {
        background-color: #fee2e2;
    }
    </style>
    """, unsafe_allow_html=True)


    # ========== TREND VISUALIZATION ==========
    st.markdown("## Tren Komentar")

    if selected_platform != "All":
        st.markdown(f"<p style='color: gray;'>Menampilkan tren komentar hanya untuk platform: <b>{selected_platform}</b></p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color: gray;'>Menampilkan tren komentar dari semua platform</p>", unsafe_allow_html=True)

    # Preprocessing untuk visualisasi
    trend_df = (
        filtered_df.groupby("Tanggal")
        .agg(
            total_comments=("standardized_text", "count"),
            negative_comments=("Sentiment", lambda x: (x == "negatif").sum())
        )
        .reset_index()
    )

    trend_df["Negativity Ratio(%)"] = (trend_df["negative_comments"] / trend_df["total_comments"]) * 100

    # Chart: Bar (Jumlah Komentar) + Line (Negativity Ratio)
    base = alt.Chart(trend_df).encode(
        x=alt.X("Tanggal:T", axis=alt.Axis(title="Tanggal", format="%d %b"))
    )

    bar = base.mark_bar(color="#facc15").encode(
        y=alt.Y("total_comments:Q", axis=alt.Axis(title="Jumlah Komentar"))
    )

    line = base.mark_line(color="#e11d48", strokeWidth=3).encode(
        y=alt.Y("Negativity Ratio(%):Q", axis=alt.Axis(title="Negativity Ratio (%)"))
    )

    points = base.mark_circle(color="#e11d48", size=60).encode(
        y=alt.Y("Negativity Ratio(%):Q")
    )

    combo_chart = alt.layer(bar, line, points).resolve_scale(
        y="independent"
    )

    st.altair_chart(combo_chart, use_container_width=True)

    st.markdown("""
    <p style='font-size: 16px; color: #333;'>
    Terlihat adanya <b>peningkatan jumlah komentar publik yang signifikan</b> menjelang <b>tanggal 25 Agustus</b>, 
    dengan <b>puncak aktivitas</b> terjadi tepat pada tanggal tersebut. 
    Hal ini mengindikasikan adanya <i>momentum atau perhatian khusus</i> dari publik terhadap acara Maybank Marathon 2024.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("## Popular Word Analysis")
    st.markdown("""
    Bagian ini menyajikan hasil analisis kata dan frasa yang paling sering digunakan dalam percakapan seputar Maybank Marathon 2024. Visualisasi berupa **wordcloud** dan **analisis n-gram** dibuat untuk menggambarkan pola umum dalam penggunaan kata oleh peserta maupun publik. Semakin sering suatu kata atau frasa muncul, maka akan semakin menonjol dalam visualisasi ini. Analisis ini bertujuan memberikan gambaran awal mengenai topik atau isu yang paling banyak dibicarakan, tanpa melihat konteks atau makna di balik kata-kata tersebut.""")

    # ========== POPULAR WORDS ANALYSIS ==========
    st.markdown("### N-Gram Analysis")

    # Clean text and filter
    def remove_emotes(text):
        return re.sub(r'\[.*?\]', '', text) if isinstance(text, str) else text

    filtered_df['text_clean'] = filtered_df['standardized_text'].apply(remove_emotes)

    # Add filters in a compact layout
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        sentiment_option = st.selectbox("Filter Sentimen:", 
                                      ["All", "positif", "netral", "negatif"], 
                                      index=0)
    with filter_col2:
        ngram_option = st.radio("Jenis N-gram:", 
                               ["Bigram", "Trigram", "Fourgram"], 
                               horizontal=True)
    with filter_col3:
        emoji_option = st.radio("Tampilkan Emoji:", 
                               ["Tanpa Emoji", "Dengan Emoji"], 
                               horizontal=True, 
                               index=0)

    # Apply filters
    words_df = filtered_df.copy()
    if sentiment_option != "All":
        words_df = words_df[words_df['Sentiment'] == sentiment_option]

    text_column = 'standardized_text' if emoji_option == "Dengan Emoji" else 'text_clean'


    # Sentimen dominan per n-gram
    def get_dominant_sentiment_ngram(ngram, df):
        hits = df[df[text_column].str.contains(re.escape(ngram), na=False)]
        if hits.empty:
            return 'netral'
        return hits['Sentiment'].value_counts().idxmax()

    # Plot with plotly - enhanced version
    def plot_ngram_chart_plotly(df, text_col, n=2):
        corpus = df[text_col].dropna().tolist()
        vec = CountVectorizer(ngram_range=(n, n), 
                             stop_words=stopwords.words('indonesian'), 
                             min_df=2)
        X = vec.fit_transform(corpus)
        ngram_freq = X.sum(axis=0).A1
        ngram_names = vec.get_feature_names_out()

        ngram_df = pd.DataFrame({'ngram': ngram_names, 'count': ngram_freq})
        ngram_df = ngram_df.sort_values('count', ascending=False).head(15)

        # Add dominant sentiment and color
        ngram_df['sentiment'] = ngram_df['ngram'].apply(lambda x: get_dominant_sentiment_ngram(x, df))
        color_map = {'positif': '#2ecc71', 'netral': '#3498db', 'negatif': '#e74c3c'}
        ngram_df['color'] = ngram_df['sentiment'].map(color_map)

        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ngram_df['ngram'][::-1],
            x=ngram_df['count'][::-1],
            orientation='h',
            marker_color=ngram_df['color'][::-1],
            hovertext=[
                f"<b>{ngram}</b><br>Frekuensi: {count}<br>Sentimen: {sent}"
                for ngram, count, sent in zip(
                    ngram_df['ngram'][::-1],
                    ngram_df['count'][::-1],
                    ngram_df['sentiment'][::-1]
                )
            ],
            hoverinfo="text"
        ))

        fig.update_layout(
            title=f"<b>Top {n}-grams</b> berdasarkan Frekuensi & Sentimen Dominan",
            xaxis_title="Frekuensi Kemunculan",
            yaxis_title="",
            yaxis=dict(tickfont=dict(size=12)),
            plot_bgcolor='white',
            height=500,
            margin=dict(l=100, r=30, t=80, b=30),
            showlegend=False,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig

    # Main visualization layout
    ngram_value = {"Bigram": 2, "Trigram": 3, "Fourgram": 4}[ngram_option]
    fig = plot_ngram_chart_plotly(words_df, text_column, n=ngram_value)
    st.plotly_chart(fig, use_container_width=True)

    # ========== WORD CLOUD SECTION ==========
    st.markdown("### Word Cloud Analysis")

    # Create tabs for each sentiment
    tab1, tab2, tab3 = st.tabs(["Positif", "Netral", "Negatif"])

    sentiment_params = [
        ("positif", "Greens"),
        ("netral", "Blues"),
        ("negatif", "Reds")
    ]

    for tab, (sentiment, colormap) in zip([tab1, tab2, tab3], sentiment_params):
        with tab:
            sentiment_text = " ".join(
                words_df[words_df['Sentiment'] == sentiment][text_column].dropna().tolist()
            )

            if sentiment_text:
                # Create two columns for word cloud and top words
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"#### Word Cloud {sentiment.capitalize()}")
                    wordcloud = WordCloud(
                        width=600,
                        height=150,
                        background_color='white',
                        colormap=colormap,
                        max_words=100
                    ).generate(sentiment_text)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

                with col2:
                    st.markdown(f"#### Top 10 Kata {sentiment.capitalize()}")
                    vec = CountVectorizer(stop_words=stopwords.words('indonesian'))
                    X = vec.fit_transform([sentiment_text])
                    word_freq = pd.DataFrame({
                        'word': vec.get_feature_names_out(),
                        'count': X.toarray()[0]
                    }).sort_values('count', ascending=False).head(10)

                    fig = px.bar(
                        word_freq,
                        x='count',
                        y='word',
                        orientation='h',
                        color='count',
                        color_continuous_scale=colormap.lower()
                    )
                    fig.update_layout(
                        showlegend=False,
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Tidak ada data untuk sentimen {sentiment}")

    # ========== TOPIC MODELLING SECTION ==========
    st.markdown("## Topic Modelling Insights")
    st.markdown("""
    Untuk mengidentifikasi tema-tema utama dalam komentar pengguna media sosial terkait Maybank Marathon 2024, dilakukan pemodelan topik (*Topic Modelling*) menggunakan pendekatan berbasis *embedding* dan *clustering*.  
    Model yang digunakan adalah **BERTopic**, yang menggabungkan representasi semantik dari teks menggunakan *transformer-based embeddings*, teknik reduksi dimensi (*UMAP*), serta algoritma *HDBSCAN* untuk mengelompokkan komentar berdasarkan kemiripan topiknya.
    """)


    # Load topic modelling data
    topic_df = pd.read_csv("data/topic_data.csv", parse_dates=["Tanggal"])
    topic_df = topic_df[topic_df['Tanggal'].notnull()]
    topic_df['Month'] = topic_df['Tanggal'].dt.to_period('M').dt.to_timestamp()

    # Pisahkan UI state dengan memberikan unique key
    sentiment_options = {
        "positif": "Positif",
        "negatif": "Negatif"
    }

    selected_sentiment = st.radio(
        "Pilih sentimen untuk analisis topik:",
        options=list(sentiment_options.keys()),
        format_func=lambda x: sentiment_options[x],
        horizontal=True,
        key="topic_sentiment_selector"
    )
    # Grouping data bulanan per topic dan sentimen
    df_grouped = (
        topic_df
        .groupby(['Month', 'Topic', 'predict_sentiment_indoBERT'])
        .size()
        .reset_index(name='Count')
    )

    df_sent = df_grouped[df_grouped['predict_sentiment_indoBERT'] == selected_sentiment]
    # Visualisasi
    if not df_sent.empty:
        fig = px.line(
            df_sent,
            x='Month',
            y='Count',
            color='Topic',
            markers=True,
            title=f"Perkembangan Topik ({selected_sentiment.capitalize()})",
            # color_discrete_sequence=px.colors.sequential.YlOrBr  # Palet kuning-oranye
            color_discrete_sequence = [
                # "#242320",  # hitam kebiruan gelap – untuk kontras
                # "#645A40",  # cokelat olive – tone bumi
                # "#FED106",  # warna utama: kuning keemasan
                # "#F4C300",  # variasi kuning lebih hangat
                "#E4D86C",  # kuning pastel
                "#B0A14A",  # emas olive tua
                "#FFE97F",  # kuning pucat / creamy
                "#D1AF1E"   # emas mustard tua
            ]

        )
        fig.update_layout(
            xaxis=dict(
                tickformat='%b %Y',
                tickangle=45,
                title='Bulan'
            ),
            yaxis_title='Frekuensi',
            legend_title='Topik',
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Tidak ada data untuk sentimen **{selected_sentiment}**.")


    with st.expander("## Actionable Insights"):
        st.markdown("""
    **Berdasarkan analisis sentimen dan topik utama terkait Maybank Marathon 2024, berikut poin-poin strategis yang dapat dilakukan:**

    1. **💬 Perkuat Komunikasi Peserta**  
    Keluhan terkait lambatnya respons & informasi tidak jelas. Solusi: tim komunikasi khusus, sistem ticketing, chatbot, dan FAQ interaktif.

    2. **🛣️ Sosialisasi Logistik & Lalu Lintas**  
    Banyak kritik soal kemacetan & penutupan jalan. Diperlukan koordinasi dengan pemda & media, serta papan info digital/fisik di lokasi strategis.

    3. **📦 Evaluasi Race Logistics**  
    Minim fasilitas seperti shuttle bus dan pencahayaan. Disarankan audit alur logistik & kesiapan medis.

    4. **🏅 Kembangkan Aspek Positif**  
    Banyak komentar positif soal atmosfer lomba & komunitas. Perluas dengan plogging, zona cheering, dan pelatihan.

    5. **📊 Manfaatkan Insight Media Sosial**  
    Percakapan tinggi saat event, turun setelahnya. Solusi: konten pasca-event (highlight, cerita peserta, teaser), serta gunakan feedback sebagai tolok ukur.
    """)


    # # app.py

    # import streamlit as st
    # from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # import torch
    # import torch.nn.functional as F

    # # Load model dan tokenizer
    # @st.cache_resource
    # def load_model():
    #     model = AutoModelForSequenceClassification.from_pretrained("saved_indobert_model")
    #     tokenizer = AutoTokenizer.from_pretrained("saved_indobert_model")
    #     model.eval()
    #     return model, tokenizer

    # model, tokenizer = load_model()

    # # Mapping label index ke nama
    # inv_label_map = {0: 'negatif', 1: 'netral', 2: 'positif'}

    # # UI
    # st.set_page_config(page_title="IndoBERT Sentiment Prediction", layout="centered")
    # st.title("🇮🇩 IndoBERT Sentiment Prediction")
    # st.markdown("Masukkan satu kalimat untuk mengetahui prediksi sentimennya (positif, netral, negatif).")

    # # Input user
    # text = st.text_area("Kalimat:", height=150)

    # if st.button("Prediksi"):
    #     if text.strip() == "":
    #         st.warning("Silakan masukkan kalimat terlebih dahulu.")
    #     else:
    #         # Tokenisasi dan prediksi
    #         inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #             probs = F.softmax(outputs.logits, dim=1)
    #             pred_label = torch.argmax(probs, dim=1).item()

    #         sentiment = inv_label_map[pred_label]
    #         st.success(f"**Prediksi Sentimen:** {sentiment.capitalize()}")

    #         # Tampilkan probabilitas per kelas
    #         st.subheader("Probabilitas:")
    #         for i, label in inv_label_map.items():
    #             st.write(f"- {label.capitalize()}: {probs[0][i].item():.2%}")


    # else:
    #     st.warning("Tidak ada data yang sesuai dengan filter yang dipilih")



# CSS styling
st.markdown("""
<style>
/* Tab styling */
[role="tab"] {
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
}

/* Section headers */
h2 {
    border-bottom: 2px solid #f0f2f6;
    padding-bottom: 8px;
    margin-top: 30px !important;
}

/* Better spacing for columns */
[data-testid="column"] {
    padding: 0 10px;
}

/* Word cloud images */
.stImage {
    margin: 0 auto;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
