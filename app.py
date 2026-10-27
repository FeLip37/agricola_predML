import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import requests
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configuração da página Streamlit
st.set_page_config(
    page_title="Plataforma de Análise Climatológica & Agrícola",
    page_icon="🌾",
    layout="wide"
)

# Dicionário de Tradução Amigável dos Lags para Termos Agronómicos
TRADUCAO_VARIAVEIS = {
    'chuva_total': 'Chuva (Mês da Colheita)',
    'radiacao_solar_total': 'Radiação Solar (Mês da Colheita)',
    'temperatura_media': 'Temperatura Média (Mês da Colheita)',
    
    'chuva_total_lag1': 'Chuva (1 Mês Antes - Maturação)',
    'chuva_total_lag2': 'Chuva (2 Meses Antes - Desenvolvimento)',
    'chuva_total_lag3': 'Chuva (3 Meses Antes - Plantio)',
    
    'radiacao_solar_total_lag1': 'Radiação Solar (1 Mês Antes - Maturação)',
    'radiacao_solar_total_lag2': 'Radiação Solar (2 Meses Antes - Desenvolvimento)',
    'radiacao_solar_total_lag3': 'Radiação Solar (3 Meses Antes - Plantio)',
    
    'temperatura_media_lag1': 'Temperatura Média (1 Mês Antes - Maturação)',
    'temperatura_media_lag2': 'Temperatura Média (2 Meses Antes - Desenvolvimento)',
    'temperatura_media_lag3': 'Temperatura Média (3 Meses Antes - Plantio)'
}

# ------------------------------------------------------------------------------
# 1. CARREGAMENTO E PROCESSAMENTO (CACHEADO VIA ST.CACHE_DATA)
# ------------------------------------------------------------------------------
@st.cache_data
def carregar_e_processar_dados():
    # Carrega a versão leve compactada em Parquet
    df = pd.read_parquet('pi_mensal.parquet')

    # Limpeza de strings e aspas residuais
    for col in ['estado', 'produto', 'mes']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.strip()

    # Tipagem numérica para colunas climáticas e de produção
    cols_num = ['quantidade_produzida', 'chuva_total', 'radiacao_solar_total', 'temperatura_media']
    for col in cols_num:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Mapeamento dos meses (Garante que a coluna mes_num é SEMPRE criada)
    meses_map = {
        'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6,
        'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12
    }
    if 'mes' in df.columns:
        df['mes_num'] = df['mes'].astype(str).map(meses_map).fillna(1).astype(int)
    else:
        df['mes_num'] = 1

    # Ordenação temporal segura
    cols_ordem = [c for c in ['estado', 'produto', 'ano', 'mes_num'] if c in df.columns]
    df = df.sort_values(by=cols_ordem).reset_index(drop=True)

    # Engenharia de Lags Climáticos
    for col in ['chuva_total', 'radiacao_solar_total', 'temperatura_media']:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df.groupby(['estado', 'produto'])[col].shift(lag)

    return df.fillna(0)

df_ml = carregar_e_processar_dados()

# ------------------------------------------------------------------------------
# 2. CÁLCULOS ESTATÍSTICOS DE SPEARMAN POR ESTADO
# ------------------------------------------------------------------------------
@st.cache_data
def calcular_sensibilidade_estados(df_input):
    cols_analise = [c for c in df_input.columns if 'lag' in c or c in ['chuva_total', 'radiacao_solar_total', 'temperatura_media']]
    
    # Otimização: Filtra apenas produtos com produção ativa
    df_com_prod = df_input[df_input['quantidade_produzida'] > 0]
    produtos_ativos = df_com_prod.groupby('estado')['produto'].unique().to_dict()

    info = {}
    for est in df_input['estado'].unique():
        prods = produtos_ativos.get(est, df_input[df_input['estado'] == est]['produto'].unique())
        max_corr_abs, melhor_prod, melhor_var, val_corr = -1, 'Sem Variação', 'N/A', 0.0

        for prod in prods:
            df_p = df_input[(df_input['estado'] == est) & (df_input['produto'] == prod)]
            if len(df_p) >= 4 and df_p['quantidade_produzida'].std() > 0:
                corrs = df_p[['quantidade_produzida'] + cols_analise].corr(method='spearman')['quantidade_produzida'].drop('quantidade_produzida').dropna()
                if not corrs.empty:
                    v_max = corrs.abs().idxmax()
                    c_val = corrs[v_max]
                    if abs(c_val) > max_corr_abs:
                        max_corr_abs = abs(c_val)
                        melhor_prod, melhor_var, val_corr = prod, v_max, c_val

        info[est] = {
            'produto_sensivel': melhor_prod,
            'variavel_critica': TRADUCAO_VARIAVEIS.get(melhor_var, melhor_var),
            'correlacao': round(val_corr, 2),
            'sensibilidade_pct': f"{abs(val_corr)*100:.1f}%"
        }
    return info

info_estados = calcular_sensibilidade_estados(df_ml)

# ------------------------------------------------------------------------------
# 3. INTERFACE PRINCIPAL STREAMLIT
# ------------------------------------------------------------------------------
st.title("🌾 Plataforma de Análise Climatológica & Agrícola do Brasil")
st.markdown("Análise espaço-temporal da relação entre clima (Open-Meteo) e safras agrícolas (SIDRA/IBGE).")

# Criar Abas da Aplicação
aba1, aba2, aba3 = st.tabs([
    "🗺️ Mapa Geográfico de Zonas", 
    "📊 Matriz de Sensibilidade", 
    "🤖 Módulo de Preparação para Machine Learning"
])

# ------------------------------------------------------------------------------
# ABA 1: MAPA DE ZONAS ESPACIAIS (GEOJSON HOVER)
# ------------------------------------------------------------------------------
with aba1:
    st.subheader("Sensibilidade Climática das Culturas Agrícolas por Estado")
    st.caption("Passe o cursor sobre qualquer estado para visualizar a cultura mais afetada e a fase climática crítica.")

    @st.cache_data
    def carregar_geojson():
        url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'
        return requests.get(url).json()

    geo_data = carregar_geojson()

    mapa_regioes = {
        'Rondônia': 'Norte', 'Acre': 'Norte', 'Amazonas': 'Norte', 'Roraima': 'Norte', 'Pará': 'Norte', 'Amapá': 'Norte', 'Tocantins': 'Norte',
        'Maranhão': 'Nordeste', 'Piauí': 'Nordeste', 'Ceará': 'Nordeste', 'Rio Grande do Norte': 'Nordeste', 'Paraíba': 'Nordeste', 'Pernambuco': 'Nordeste', 'Alagoas': 'Nordeste', 'Sergipe': 'Nordeste', 'Bahia': 'Nordeste',
        'Minas Gerais': 'Sudeste', 'Espírito Santo': 'Sudeste', 'Rio de Janeiro': 'Sudeste', 'São Paulo': 'Sudeste',
        'Paraná': 'Sul', 'Santa Catarina': 'Sul', 'Rio Grande do Sul': 'Sul',
        'Mato Grosso do Sul': 'Centro-Oeste', 'Mato Grosso': 'Centro-Oeste', 'Goiás': 'Centro-Oeste', 'Distrito Federal': 'Centro-Oeste'
    }
    cores_regiao = {'Norte': '#2ca02c', 'Nordeste': '#ff7f0e', 'Centro-Oeste': '#9467bd', 'Sudeste': '#d62728', 'Sul': '#1f77b4'}

    # Normalizador de texto para correspondência perfeita entre GeoJSON e DataFrame
    def norm_text(txt):
        if not isinstance(txt, str): return ""
        return unicodedata.normalize('NFD', txt).encode('ascii', 'ignore').decode('utf-8').strip().lower()

    info_norm = {norm_text(k): v for k, v in info_estados.items()}

    for feature in geo_data['features']:
        nome_est = feature['properties']['name']
        reg = mapa_regioes.get(nome_est, 'Norte')
        st_info = info_norm.get(norm_text(nome_est), {
            'produto_sensivel': 'Dados insuficientes', 'variavel_critica': 'N/A', 'correlacao': 0.0, 'sensibilidade_pct': '0%'
        })
        
        feature['properties']['regiao'] = reg
        feature['properties']['cor_regiao'] = cores_regiao.get(reg, '#cccccc')
        feature['properties']['produto'] = st_info['produto_sensivel']
        feature['properties']['variavel'] = st_info['variavel_critica']
        feature['properties']['correlacao'] = str(st_info['correlacao'])
        feature['properties']['sensibilidade'] = st_info['sensibilidade_pct']

    # Mapa centrado no Brasil (Camada Esri)
    mapa = folium.Map(
        location=[-14.2350, -51.9253], 
        zoom_start=4, 
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', 
        attr='Esri'
    )

    folium.GeoJson(
        geo_data,
        style_function=lambda f: {
            'fillColor': f['properties']['cor_regiao'], 
            'color': 'white', 
            'weight': 1.5, 
            'fillOpacity': 0.50
        },
        highlight_function=lambda f: {
            'fillColor': f['properties']['cor_regiao'], 
            'color': 'black', 
            'weight': 3, 
            'fillOpacity': 0.85
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'regiao', 'produto', 'variavel', 'correlacao', 'sensibilidade'],
            aliases=['Estado:', 'Região:', 'Cultura Mais Afetada:', 'Fase Climática Crítica:', 'Índice Spearman:', 'Sensibilidade Climática:'],
            sticky=True,
            style="background-color: #ffffff; border: 2px solid #333333; border-radius: 6px; padding: 10px; font-family: Arial, sans-serif; font-size: 13px;"
        )
    ).add_to(mapa)

    st_folium(mapa, width="100%", height=550)

# ------------------------------------------------------------------------------
# ABA 2: TABELA RESUMO DE SENSIBILIDADE ESTATÍSTICA
# ------------------------------------------------------------------------------
with aba2:
    st.subheader("Resumo Estatístico da Correlação de Spearman por Estado")
    df_resumo = pd.DataFrame.from_dict(info_estados, orient='index')
    df_resumo.columns = ['Cultura Mais Afetada', 'Fase Climática Crítica', 'Correlação (Spearman)', 'Sensibilidade Climática (%)']
    st.dataframe(df_resumo, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 3: MÓDULO DE PREPARAÇÃO PARA MACHINE LEARNING (REPLIT / DEPLOY)
# ------------------------------------------------------------------------------
with aba3:
    st.subheader("Módulo de Engenharia de Recursos para Machine Learning")
    st.info("Esta secção estrutura as matrizes de entrada (X) e a variável alvo (y) com pré-processamento para integração no Replit ou APIs, sem executar treino de modelos nesta fase.")

    colunas_climaticas = [c for c in df_ml.columns if 'lag' in c or c in ['chuva_total', 'radiacao_solar_total', 'temperatura_media']]
    colunas_cat = ['estado', 'produto', 'mes_num']

    df_ml_ready = df_ml[df_ml['quantidade_produzida'] > 0].copy()
    X = df_ml_ready[colunas_cat + colunas_climaticas]
    y = df_ml_ready['quantidade_produzida']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), colunas_climaticas),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colunas_cat)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    c1, c2 = st.columns(2)
    c1.metric("Amostras de Treino (X_train)", f"{X_train.shape[0]:,} linhas")
    c2.metric("Amostras de Teste (X_test)", f"{X_test.shape[0]:,} linhas")

    st.write("Amostra da Matriz de Entrada (Features X_train):")
    st.dataframe(X_train.head(10), use_container_width=True)

    st.write("Amostra do Alvo de Produção (Target y_train):")
    st.dataframe(y_train.head(10), use_container_width=True)

    st.subheader("Exportar Ficheiros de Dados Prontos para o Replit")
    st.download_button(
        label="📥 Baixar X_train_agricola.csv",
        data=X_train.to_csv(index=False),
        file_name="X_train_agricola.csv",
        mime="text/csv"
    )
    st.download_button(
        label="📥 Baixar y_train_agricola.csv",
        data=y_train.to_csv(index=False),
        file_name="y_train_agricola.csv",
        mime="text/csv"
    )