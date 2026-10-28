import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import requests
import unicodedata
import os
import gc
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------------------------
icone_path = "icon.png" if os.path.exists("icon.png") else None

st.set_page_config(
    page_title="Plataforma de Análise Climatológica e Agrícola",
    page_icon=icone_path if icone_path else "circle",
    layout="wide"
)

# ------------------------------------------------------------------------------
# INJEÇÃO CSS: TEMA CLARO CORPORATIVO & FILTRO FLUTUANTE SOBRE O MAPA
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* FORÇAR TEMA CLARO E FUNDO BRANCO */
    html, body, [data-testid="stAppViewContainer"], .stApp, main, header, section {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label, caption {
        color: #111827 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stCaption, p {
        color: #4B5563 !important;
    }

    /* ABAS DE NAVEGAÇÃO EM VERDE AGRO (#15803D) */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        padding: 10px 16px !important;
    }

    button[data-baseweb="tab"] * {
        color: #4B5563 !important;
        font-weight: 500 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] * {
        color: #15803D !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #15803D !important;
    }

    div[data-baseweb="tab-border"] {
        background-color: #E5E7EB !important;
    }

    /* BOTÕES E MÉTRICAS */
    .stButton>button, .stDownloadButton>button {
        background-color: #15803D !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        border: none !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #166534 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stMetricValue"] {
        color: #15803D !important;
    }

    .stAlert {
        background-color: #F9FAFB !important;
        color: #374151 !important;
        border: 1px solid #E5E7EB !important;
        border-left: 4px solid #15803D !important;
    }

    /* CONTAINER DO MAPA COM OVERLAY FLUTUANTE */
    .map-container-relative {
        position: relative;
        width: 100%;
    }

    /* CAIXA DE FILTRO FLUTUANTE SOBRE O MAPA */
    .map-floating-card {
        background: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(4px);
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TRADUÇÃO DAS VARIÁVEIS
# ------------------------------------------------------------------------------
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
# 1. CARREGAMENTO COM OTIMIZAÇÃO DE MEMÓRIA (DOWNCASTING)
# ------------------------------------------------------------------------------
@st.cache_data
def carregar_e_processar_dados():
    cols_para_carregar = [
        'estado', 'produto', 'ano', 'mes', 
        'quantidade_produzida', 'chuva_total', 
        'radiacao_solar_total', 'temperatura_media'
    ]
    
    try:
        df = pd.read_parquet('pi_mensal.parquet', columns=cols_para_carregar)
    except Exception:
        df = pd.read_parquet('pi_mensal.parquet')

    for col in ['estado', 'produto', 'mes']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.strip()

    cols_num = ['quantidade_produzida', 'chuva_total', 'radiacao_solar_total', 'temperatura_media']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)

    if 'ano' in df.columns:
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce').fillna(2000).astype(np.int16)

    meses_map = {
        'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6,
        'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12
    }
    if 'mes' in df.columns:
        df['mes_num'] = df['mes'].astype(str).map(meses_map).fillna(1).astype(np.int8)
    else:
        df['mes_num'] = np.int8(1)

    cols_ordem = [c for c in ['estado', 'produto', 'ano', 'mes_num'] if c in df.columns]
    df = df.sort_values(by=cols_ordem).reset_index(drop=True)

    for col in ['chuva_total', 'radiacao_solar_total', 'temperatura_media']:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df.groupby(['estado', 'produto'])[col].shift(lag).astype(np.float32)

    df = df.fillna(0)
    gc.collect()
    return df

df_ml = carregar_e_processar_dados()

# ------------------------------------------------------------------------------
# 2. PRÉ-CÁLCULO TOTAL DAS CORRELAÇÕES (ZERO LAG)
# ------------------------------------------------------------------------------
@st.cache_data
def precalcular_todas_sensibilidades(df_input):
    cols_analise = [c for c in df_input.columns if 'lag' in c or c in ['chuva_total', 'radiacao_solar_total', 'temperatura_media']]
    df_valid = df_input[df_input['quantidade_produzida'] > 0]
    
    sensib_map = {}
    visao_geral = {}

    grouped = df_valid.groupby(['estado', 'produto'])
    
    for (est, prod), group in grouped:
        if len(group) >= 4 and group['quantidade_produzida'].std() > 0:
            corrs = group[['quantidade_produzida'] + cols_analise].corr(method='spearman')['quantidade_produzida'].drop('quantidade_produzida').dropna()
            if not corrs.empty:
                v_max = corrs.abs().idxmax()
                c_val = float(corrs[v_max])
                abs_c = abs(c_val)

                item = {
                    'produto_sensivel': prod,
                    'variavel_critica': TRADUCAO_VARIAVEIS.get(v_max, v_max),
                    'correlacao': round(c_val, 2),
                    'sensibilidade_pct': f"{abs_c*100:.1f}%",
                    'abs_corr': abs_c
                }

                if prod not in sensib_map:
                    sensib_map[prod] = {}
                sensib_map[prod][est] = item

                if est not in visao_geral or abs_c > visao_geral[est]['abs_corr']:
                    visao_geral[est] = item

    sensib_map["Visão Geral (Cultura Mais Sensível por Estado)"] = visao_geral
    return sensib_map

matriz_sensibilidade = precalcular_todas_sensibilidades(df_ml)

# ------------------------------------------------------------------------------
# CABEÇALHO DO DASHBOARD
# ------------------------------------------------------------------------------
def get_image_base64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

img_b64 = get_image_base64(icone_path)

if img_b64:
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 6px;">
            <img src="data:image/png;base64,{img_b64}" style="width: 44px; height: 44px; object-fit: contain;" />
            <h1 style="margin: 0; padding: 0; font-size: 1.75rem; font-weight: 700; color: #111827 !important;">Plataforma de Análise Climatológica e Agrícola do Brasil</h1>
        </div>
        <p style="margin: 0 0 16px 0; color: #4B5563 !important; font-size: 0.92rem;">Análise espaço-temporal e modelagem estatística entre dados meteorológicos (Open-Meteo) e safras agrícolas (IBGE/SIDRA).</p>
    """, unsafe_allow_html=True)
else:
    st.markdown('<h1 style="color: #111827 !important;">Plataforma de Análise Climatológica e Agrícola do Brasil</h1>', unsafe_allow_html=True)
    st.caption("Análise espaço-temporal e modelagem estatística entre dados meteorológicos (Open-Meteo) e safras agrícolas (IBGE/SIDRA).")

st.divider()

# ------------------------------------------------------------------------------
# ABAS DE NAVEGAÇÃO
# ------------------------------------------------------------------------------
aba1, aba2, aba3 = st.tabs([
    "Mapa Geográfico de Zonas", 
    "Matriz de Sensibilidade", 
    "Módulo de Preparação para Machine Learning"
])

# ------------------------------------------------------------------------------
# ABA 1: MAPA GEOGRÁFICO DE ZONAS COM FILTRO INTEGRADO
# ------------------------------------------------------------------------------
with aba1:
    st.markdown('<h3 style="color: #111827 !important;">Sensibilidade Climática das Culturas Agrícolas por Estado</h3>', unsafe_allow_html=True)

    # BARRA/CARD DE CONTROLE DO MAPA (ESTILO PAINEL INTEGRADO)
    opcoes_filtro = ["Visão Geral (Cultura Mais Sensível por Estado)"] + sorted([k for k in matriz_sensibilidade.keys() if k != "Visão Geral (Cultura Mais Sensível por Estado)"])

    st.markdown('<div class="map-floating-card">', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns([0.45, 0.55])
    
    with col_f1:
        produto_selecionado = st.selectbox(
            "Filtrar cultura no mapa:",
            opcoes_filtro,
            label_visibility="visible"
        )

    info_estados = matriz_sensibilidade.get(produto_selecionado, {})

    destaque = None
    if produto_selecionado != "Visão Geral (Cultura Mais Sensível por Estado)" and info_estados:
        est_max = max(info_estados.items(), key=lambda x: x[1].get('abs_corr', 0), default=(None, None))
        if est_max[0] and est_max[1].get('abs_corr', 0) > 0:
            destaque = (est_max[0], est_max[1])

    with col_f2:
        if destaque:
            st.info(f"**Maior Impacto em {produto_selecionado}:** {destaque[0]} "
                    f"({destaque[1]['sensibilidade_pct']} - {destaque[1]['variavel_critica']})")
        else:
            st.caption("Passe o cursor sobre a área de qualquer estado para visualizar os detalhes climáticos.")
    st.markdown('</div>', unsafe_allow_html=True)

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
    cores_regiao = {'Norte': '#15803D', 'Nordeste': '#D97706', 'Centro-Oeste': '#7C3AED', 'Sudeste': '#DC2626', 'Sul': '#2563EB'}

    def norm_text(txt):
        if not isinstance(txt, str): return ""
        return unicodedata.normalize('NFD', txt).encode('ascii', 'ignore').decode('utf-8').strip().lower()

    info_norm = {norm_text(k): v for k, v in info_estados.items()}

    for feature in geo_data['features']:
        nome_est = feature['properties']['name']
        reg = mapa_regioes.get(nome_est, 'Norte')
        st_info = info_norm.get(norm_text(nome_est), {
            'produto_sensivel': 'Sem dados suficientes', 'variavel_critica': 'N/A', 'correlacao': 0.0, 'sensibilidade_pct': '0%'
        })
        
        feature['properties']['regiao'] = reg
        feature['properties']['cor_regiao'] = cores_regiao.get(reg, '#CCCCCC')
        feature['properties']['produto'] = st_info['produto_sensivel']
        feature['properties']['variavel'] = st_info['variavel_critica']
        feature['properties']['correlacao'] = str(st_info['correlacao'])
        feature['properties']['sensibilidade'] = st_info['sensibilidade_pct']

    # MAPA COM VISUALIZAÇÃO GLOBAL SUAVE (ESRI LIGHT GRAY CANVAS - SEM API KEY)
    mapa = folium.Map(
        location=[-14.2350, -51.9253], 
        zoom_start=4,
        tiles=None,
        zoom_control=True
    )

    # Camada de mapa leve em tons claros mantendo países vizinhos visíveis
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        name='Esri Light Gray Canvas',
        attr='Esri, HERE, Garmin, USGS, NGA, EPA, USDA, NPS'
    ).add_to(mapa)

    folium.GeoJson(
        geo_data,
        style_function=lambda f: {
            'fillColor': f['properties']['cor_regiao'], 
            'color': '#FFFFFF', 
            'weight': 1.5, 
            'fillOpacity': 0.65
        },
        highlight_function=lambda f: {
            'fillColor': f['properties']['cor_regiao'], 
            'color': '#111827', 
            'weight': 2.5, 
            'fillOpacity': 0.