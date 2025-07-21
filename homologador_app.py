
import streamlit as st
import pandas as pd
import difflib

# Cargar datos históricos de homologación
df_base = pd.read_excel("test2.xlsx")
df_base.columns = df_base.columns.str.strip()
homologadas = df_base['Homologado'].dropna().unique().tolist()

# Subir archivo con nuevas entradas
st.title("App de Homologación Manual Asistida")
archivo_nuevo = st.file_uploader("Sube el archivo con nuevas entradas a homologar", type=["csv", "xlsx"])

if archivo_nuevo:
    if archivo_nuevo.name.endswith(".csv"):
        df_nuevo = pd.read_csv(archivo_nuevo)
    else:
        df_nuevo = pd.read_excel(archivo_nuevo)

    df_nuevo.columns = df_nuevo.columns.str.strip()
    columna_entrada = st.selectbox("Selecciona la columna con valores a homologar", df_nuevo.columns)

    # Calcular sugerencia de etiqueta
    resultados = []
    for val in df_nuevo[columna_entrada].dropna().unique():
        match = difflib.get_close_matches(val, homologadas, n=1, cutoff=0)
        mejor_match = match[0] if match else ""
        score = difflib.SequenceMatcher(None, val, mejor_match).ratio() if match else 0
        resultados.append({
            "entrada": val,
            "sugerido": mejor_match,
            "similitud": round(score, 3),
            "confirmado": ""
        })

    df_resultado = pd.DataFrame(resultados)
    st.subheader("Resultados de clasificación (similitud < 0.9)")
    df_filtrado = df_resultado[df_resultado['similitud'] < 0.9].reset_index(drop=True)

    for idx, row in df_filtrado.iterrows():
        st.write(f"**Entrada:** {row['entrada']} (sugerido: {row['sugerido']}, similitud: {row['similitud']})")
        df_filtrado.at[idx, "confirmado"] = st.text_input(f"Ingrese clasificación final para '{row['entrada']}'", value=row["sugerido"], key=f"input_{idx}")

    if st.button("Guardar resultado en CSV"):
        df_filtrado.to_csv("homologaciones_clasificadas.csv", index=False)
        st.success("Archivo guardado como 'homologaciones_clasificadas.csv'")
