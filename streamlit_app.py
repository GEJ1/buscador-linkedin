import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pytrends.request import TrendReq

st.markdown("""
<div style="text-align: center;">
    <h1 style="font-size: 2.5em; margin-bottom: 0.5em;">🔍 Búsqueda Semántica de Publicaciones en LinkedIn</h1>
    <h2 style="margin-top: 0.5em; font-weight: normal;">
        <a href="https://www.linkedin.com/in/gustavo-juantorena/" 
           target="_blank" 
           style="color: #0073b1; text-decoration: none; margin-right: 20px;">
           🌐 Seguime en LinkedIn
        </a>
        <a href="https://github.com/gej1" 
           target="_blank" 
           style="color: black; text-decoration: none;">
           <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
                alt="GitHub" 
                style="width: 30px; vertical-align: middle; margin-right: 8px;">
           Mi GitHub
        </a>
    </h2>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def get_google_trends(query):
    """Obtiene datos de tendencias de Google Trends."""
    pytrends = TrendReq(hl='es-ES', tz=360)
    pytrends.build_payload([query], cat=0, timeframe='today 12-m', geo='', gprop='')
    data = pytrends.interest_over_time()
    if not data.empty:
        return data.reset_index()
    return None


# Cargar datos
@st.cache_data
def load_data(file_path):
    """Carga el archivo CSV con publicaciones y embeddings."""
    try:
        data = pd.read_csv(file_path)
        data['ShareCommentary'] = data['ShareCommentary'].fillna("")
        data = data[data['ShareCommentary'].str.len() > 100]  # Filtrar publicaciones cortas

        # Convertir embeddings a listas
        data['embeddings'] = data['embeddings'].apply(json.loads)
        return data
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

# Cargar modelo de embeddings
@st.cache_resource
def load_model(model_name):
    """Carga el modelo de SentenceTransformer."""
    return SentenceTransformer(model_name)

# Calcular similitud
def get_top_similar_posts(query, data, embeddings, model, top_n):
    """Calcula las publicaciones más similares a la consulta."""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    data['similarity'] = similarities
    return data.sort_values(by="similarity", ascending=False).head(top_n)

# Ruta del archivo de datos
file_path = "data/Shares_with_embeddings_sentence_similarity_spanish_es_JSON.csv"


# Cargar datos y modelo
with st.spinner("Cargando datos y modelo..."):
    data = load_data(file_path)
    embeddings = np.vstack(data['embeddings'].values) if not data.empty else None
    model = load_model("hiiamsid/sentence_similarity_spanish_es")

# Input del usuario
# Formulario para búsqueda con soporte para Enter
with st.form("search_form"):
    query = st.text_input("Escribe tu consulta para buscar publicaciones:")
    submit_button = st.form_submit_button("Buscar")
top_n = st.slider("¿Cuántos resultados deseas mostrar?", min_value=1, max_value=20, value=5)

# Botón de búsqueda
if submit_button and query:
    if query and not data.empty and embeddings is not None:
        # Obtener resultados
        results = get_top_similar_posts(query, data, embeddings, model, top_n)
            # Mostrar el gráfico de Google Trends pequeño
        try:
            trends_data = get_google_trends(query)
        except:
            pass
        if trends_data is not None:
            st.write("### Tendencias de Google para tu consulta:")
            st.line_chart(trends_data.set_index('date')[[query]], height=150)
        else:
            st.warning("No se encontraron datos de tendencias para esta consulta.")

        st.markdown("""
        ### Posts de Linkedin más relevantes""")


        for idx, row in results.iterrows():
            st.write(f"#### Publicación {idx + 1}")
            st.write(row['ShareCommentary'][:200])  # Mostrar resumen
            with st.expander("Ver texto completo"):
                st.write(row['ShareCommentary'])

            # Botón para el enlace
            link = row.get('ShareLink')
            if pd.notna(link):
                st.markdown(
                    f"""
                    <a href="{link}" target="_blank" style="
                        display: inline-block;
                        text-decoration: none;
                        border-radius: 5px;
                        background-color: white;
                        color: 0073b1;
                        padding: 8px 16px;
                        border-radius: 5px;
                        font-weight: bold;
                    ">
                    🌐 Ver Publicación Completa en LinkedIn
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
   

    else:
        if not query:
            st.info("Por favor, escribe una consulta para comenzar.")
        elif data.empty:
            st.warning("No se encontraron datos para cargar.")
