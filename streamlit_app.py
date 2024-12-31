import streamlit as st
import pandas as pd
import math
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


# Configurar título de la aplicación
st.title("🔍 Búsqueda Semántica de Publicaciones en LinkedIn")

# Cargar los datos
@st.cache_data
def load_data():
    # Cargar el CSV con las publicaciones y embeddings
    data = pd.read_csv("Shares_with_embeddings_sentence_similarity_spanish_es.csv")
    return data

data = load_data()

# Cargar el modelo de SentenceTransformer
@st.cache_resource
def load_model():
    return SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")

model = load_model()

# Input de búsqueda
query = st.text_input("Escribe tu consulta para buscar publicaciones:")

if query:
    # Generar el embedding de la consulta
    query_embedding = model.encode([query])
    embeddings = np.vstack(data['embeddings'])
    
    # Calcular la similaridad entre la consulta y las publicaciones
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    data['similarity'] = similarities
    
    # Ordenar las publicaciones por relevancia
    results = data.sort_values(by="similarity", ascending=False).head(5)

    # Mostrar resultados en modales con iframes
    for i, row in results.iterrows():
        st.write(f"### {row['ShareCommentary'][:100]}...")  # Muestra un fragmento del post
        if st.button(f"Ver Publicación Completa {i+1}", key=f"modal_{i}"):
            # Modal con iframe
            st.markdown(
                f"""
                <iframe src="{row['ShareLink']}" width="800" height="600" frameborder="0"></iframe>
                """,
                unsafe_allow_html=True,
            )
