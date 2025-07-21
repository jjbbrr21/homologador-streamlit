import streamlit as st
import pandas as pd
import difflib
import unidecode
from sentence_transformers import SentenceTransformer, util

# ========== CONFIG ========== #
st.title("App de Homologación Inteligente")

# Cargar base histórica
df_base = pd.read_excel("test2.xlsx")
df_base.columns = df_base.columns.str.strip()
homologadas = df_base['Homologado'].dropna().unique().tolist()

# Modelo de embeddings
modelo = SentenceTransformer('all-MiniLM-L6-v2')
homologadas_emb = modelo.encode(homologadas, convert_to_tensor=True)

# Funciones
def limpiar_texto(texto):
    texto = unidecode.unidecode(str(texto).lower().strip())
    palabras = texto.split()
    palabras = [p for p in palabras if p not in ['la', 'el', 'de', 'del', 'los', 'las']]
    return ' '.join(palabras)

def encontrar_mejor_candidato(entrada_raw):
    entrada = limpiar_texto(entrada_raw)

    # Similitud difflib
    match_dif = difflib.get_close_matches(entrada, homologadas, n=1, cutoff=0)
    mejor_match_dif = match_dif[0] if match_dif else ""
    score_dif = difflib.SequenceMatcher(None, entrada, mejor_match_dif).ratio() if match_dif else 0

    # Similitud semántica (embeddings)
    emb_entrada = modelo.encode(entrada, convert_to_tensor=True)
    score_coseno = util.pytorch_cos_sim(emb_entrada, homologadas_emb)[0]
    idx_mejor = score_coseno.argmax().item()
    score_sem = score_coseno[idx_mejor].item()
    mejor_match_sem = homologadas[idx_mejor]

    # Normalización + heurística
    if limpiar_texto(mejor_match_dif) == entrada:
        score_heur = 1.0
    else:
        score_heur = 0

    # Puntaje combinado (ajustable)
    score_total = (score_dif * 0.4) + (score_sem * 0.5) + (score_heur * 0.1)

    if score_total < 0.9:
        return "NO SE ENCONTRO CANDIDATO OPTIMO", round(score_total, 3)
    else:
        return mejor_match_sem, round(score_total, 3)

# ========== APP STREAMLIT ========== #
archivo_nuevo = st.file_uploader("Sube el archivo con nuevas entradas", type=["csv", "xlsx"])

if archivo_nuevo:
    if archivo_nuevo.name.endswith(".csv"):
        df_nuevo = pd.read_csv(archivo_nuevo)
    else:
        df_nuevo = pd.read_excel(archivo_nuevo)

    df_nuevo.columns = df_nuevo.columns.str.strip()
    columna_entrada = st.selectbox("Selecciona la columna con valores a homologar", df_nuevo.columns)

    resultados = []
    for val in df_nuevo[columna_entrada].dropna().unique():
        sugerido, score = encontrar_mejor_candidato(val)
        resultados.append({
            "entrada": val,
            "sugerido": sugerido,
            "similitud_total": score,
            "confirmado": ""
        })

    df_resultado = pd.DataFrame(resultados)
    st.subheader("Resultados")
    df_resultado = df_resultado.sort_values(by='similitud_total', ascending=False).reset_index(drop=True)

    for idx, row in df_resultado.iterrows():
        st.write(f"**Entrada:** {row['entrada']} → Sugerido: _{row['sugerido']}_ (similitud: {row['similitud_total']})")
        df_resultado.at[idx, "confirmado"] = st.text_input(f"Clasificación final para '{row['entrada']}'", value=row["sugerido"], key=f"input_{idx}")

    if st.button("Guardar resultado en CSV"):
        df_resultado.to_csv("homologaciones_clasificadas.csv", index=False)
        st.success("Archivo guardado como 'homologaciones_clasificadas.csv'")
