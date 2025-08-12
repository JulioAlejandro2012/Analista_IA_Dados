import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()
MODEL = os.getenv("LLM_MODEL", "llama2:7b-chat-q2_K")

# Inicializa LLM
llm = OllamaLLM(model=MODEL)

# Função para análise
def analisar_dataset(df):
    resumo = {}
    resumo["Linhas e Colunas"] = df.shape
    resumo["Colunas"] = list(df.columns)
    resumo["Tipos de Dados"] = df.dtypes.astype(str).to_dict()
    resumo["Valores Nulos"] = df.isnull().sum().to_dict()
    resumo["Estatísticas"] = df.describe().to_dict()

    # Gera o gráfico de valores ausentes
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    chart_path = "charts/missing_values.png"
    os.makedirs("charts", exist_ok=True)
    plt.savefig(chart_path)
    plt.close()

    return resumo, chart_path

# Função para gerar explicação
def gerar_explicacao(resumo):
    prompt = f"""
    Você é um analista de dados. Analise o seguinte resumo estatístico de um dataset
    e escreva um texto com os principais insights, porém de forma objetiva e resumida.
    Escreva apenas coisas importantes e úteis, em português simples:

    {resumo}
    """
    return llm.invoke(prompt)

# Interface Streamlit
st.title("📊 Analista de Dados com IA Explicadora")
uploaded_file = st.file_uploader("Envie seu arquivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Prévia dos Dados")
    st.write(df.head())

    resumo, chart_path = analisar_dataset(df)
    st.subheader("Resumo Estatístico")
    st.json(resumo)

    st.image(chart_path, caption="Mapa de valores ausentes")

    st.subheader("Explicação da IA")
    explicacao = gerar_explicacao(resumo)
    st.write(explicacao)
