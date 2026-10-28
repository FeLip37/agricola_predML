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
# INJEÇÃO CSS: TEMA CLARO CORPORATIVO E SELECTBOX LEGÍVEL
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], .stApp, main, header, section {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }

    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2rem !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label, caption {
        color: #111827 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* CONTRASTE PARA O MENU SUSPENSO (SELECTBOX) */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-color: #D1D5DB !important;
    }

    div[data-baseweb="popover"], div[role="listbox"], ul[role="listbox"] {
        background-color: #FFFFFF !important;
    }

    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
    }

    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #F3F4F6 !important;
        color: #15803D !important;
    }

    /* ABAS EM VERDE AGRO (#15803D) */
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
# 1. CARREGAMENTO E PROCESSAMENTO DE DADOS
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
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    meses_map = {
        'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6,
        'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12
    }
    if 'mes' in df.columns:
        df['mes_num'] = df['mes'].astype(str).map(meses_map).fillna(1).astype(int)
    else:
        df['mes_num'] = 1

    cols_ordem = [c for c in ['estado', 'produto', 'ano', 'mes_num'] if c in df.columns]
    df = df.sort_values(by=cols_ordem).reset_index(drop=True)

    for col in ['chuva_total', 'radiacao_solar_total', 'temperatura_media']:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df.groupby(['estado', 'produto'])[col].shift(lag)

    return df.fillna(0)

df_ml = carregar_e_processar_dados()

# ------------------------------------------------------------------------------
# 2. CÁLCULO DE SPEARMAN ORIGINAL (LÓGICA ANTERIOR RESTAURADA)
# ------------------------------------------------------------------------------
@st.cache_data
def calcular_sensibilidade_geral(df_input):
    cols_analise = [c for c in df_input.columns if 'lag' in c or c in ['chuva_total', 'radiacao_solar_total', 'temperatura_media']]
    
    # Restaura exatamente a lógica original que o usuário validou
    df_com_prod = df_input[df_input['quantidade_produzida'] > 0]
    produtos_ativos = df_com_prod.groupby('estado')['produto'].unique().to_dict()

    sensib_map = {}
    visao_geral = {}

    # Pré-calcula para todos os produtos e visão geral
    todos_produtos = df_input['produto'].unique()
    
    for prod in todos_produtos:
        sensib_map[prod] = {}
        for est in df_input['estado'].unique():
            df_p = df_input[(df_input['estado'] == est) & (df_input['produto'] == prod)]
            if len(df_p) >= 4 and df_p['quantidade_produzida'].std() > 0:
                corrs = df_p[['quantidade_produzida'] + cols_analise].corr(method='spearman')['quantidade_produzida'].drop('quantidade_produzida').dropna()
                if not corrs.empty:
                    v_max = corrs.abs().idxmax()
                    c_val = float(corrs[v_max])
                    abs_c = abs(c_val)
                    sensib_map[prod][est] = {
                        'produto_sensivel': prod,
                        'variavel_critica': TRADUCAO_VARIAVEIS.get(v_max, v_max),
                        'correlacao': round(c_val, 2),
                        'sensibilidade_pct': f"{abs_c*100:.1f}%",
                        'abs_corr': abs_c
                    }

    # Visão geral (cultura mais sensível por estado)
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

        visao_geral[est] = {
            'produto_sensivel': melhor_prod,
            'variavel_critica': TRADUCAO_VARIAVEIS.get(melhor_var, melhor_var),
            'correlacao': round(float(val_corr), 2),
            'sensibilidade_pct': f"{abs(float(val_corr))*100:.1f}%",
            'abs_corr': abs(float(val_corr))
        }

    sensib_map["Visão Geral (Cultura Mais Sensível por Estado)"] = visao_geral
    return sensib_map

matriz_sensibilidade = calcular_sensibilidade_geral(df_ml)

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
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 4px;">
            <img src="data:image/png;base64,{img_b64}" style="width: 42px; height: 42px; object-fit: contain;" />
            <h1 style="margin: 0; padding: 0; font-size: 1.7rem; font-weight: 700; color: #111827 !important;">Plataforma de Análise Climatológica e Agrícola do Brasil</h1>
        </div>
        <p style="margin: 0 0 14px 0; color: #4B5563 !important; font-size: 0.90rem;">Análise espaço-temporal e modelagem estatística entre dados meteorológicos (Open-Meteo) e safras agrícolas (IBGE/SIDRA).</p>
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
# ABA 1: MAPA GEOGRÁFICO DE ZONAS
# ------------------------------------------------------------------------------
with aba1:
    st.markdown('<h3 style="color: #111827 !important; margin-bottom: 4px;">Sensibilidade Climática das Culturas Agrícolas por Estado</h3>', unsafe_allow_html=True)

    opcoes_filtro = ["Visão Geral (Cultura Mais Sensível por Estado)"] + sorted([k for k in matriz_sensibilidade.keys() if k != "Visão Geral (Cultura Mais Sensível por Estado)"])

    col_sel, col_info = st.columns([0.40, 0.60])
    with col_sel:
        produto_selecionado = st.selectbox(
            "Filtrar cultura no mapa:",
            opcoes_filtro
        )

    info_estados = matriz_sensibilidade.get(produto_selecionado, {})

    destaque = None
    if produto_selecionado != "Visão Geral (Cultura Mais Sensível por Estado)" and info_estados:
        est_max = max(info_estados.items(), key=lambda x: x[1].get('abs_corr', 0), default=(None, None))
        if est_max[0] and est_max[1].get('abs_corr', 0) > 0:
            destaque = (est_max[0], est_max[1])

    with col_info:
        if destaque:
            st.info(f"**Maior Impacto em {produto_selecionado}:** {destaque[0]} "
                    f"({destaque[1]['sensibilidade_pct']} - {destaque[1]['variavel_critica']})")
        else:
            st.caption("Passe o cursor sobre a área de qualquer estado para visualizar os detalhes climáticos.")

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

    mapa = folium.Map(
        location=[-14.2350, -51.9253], 
        zoom_start=4,
        tiles=None,
        zoom_control=True
    )

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
            'fillOpacity': 0.85
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'regiao', 'produto', 'variavel', 'correlacao', 'sensibilidade'],
            aliases=['Estado:', 'Região:', 'Cultura Analisada:', 'Fase Climática Crítica:', 'Índice Spearman:', 'Sensibilidade Climática:'],
            sticky=True,
            style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 10px; font-family: Inter, sans-serif; font-size: 13px; color: #1F2937;"
        )
    ).add_to(mapa)

    st_folium(mapa, width="100%", height=530, returned_objects=[])

# ------------------------------------------------------------------------------
# ABA 2: MATRIZ DE SENSIBILIDADE
# ------------------------------------------------------------------------------
with aba2:
    st.markdown('<h3 style="color: #111827 !important;">Resumo Estatístico da Correlação de Spearman por Estado</h3>', unsafe_allow_html=True)
    df_resumo = pd.DataFrame.from_dict(info_estados, orient='index')
    if not df_resumo.empty and 'abs_corr' in df_resumo.columns:
        df_resumo = df_resumo.drop(columns=['abs_corr'])
    if not df_resumo.empty:
        df_resumo.columns = ['Cultura Analisada', 'Fase Climática Crítica', 'Correlação (Spearman)', 'Sensibilidade Climática (%)']
    st.dataframe(df_resumo, use_container_width=True)

# ------------------------------------------------------------------------------
# ABA 3: MÓDULO DE PREPARAÇÃO PARA MACHINE LEARNING
# ------------------------------------------------------------------------------
with aba3:
    st.markdown('<h3 style="color: #111827 !important;">Módulo de Engenharia de Recursos para Machine Learning</h3>', unsafe_allow_html=True)
    st.info("Estruturação das matrizes de entrada (X) e alvo (y) com pré-processamento para integração posterior via Replit ou API externa.")

    colunas_climaticas = [c for c in df_ml.columns if 'lag' in c or c in ['chuva_total', 'radiacao_solar_total', 'temperatura_media']]
    colunas_cat = ['estado', 'produto', 'mes_num']

    df_ml_ready = df_ml[df_ml['quantidade_produzida'] > 0].copy()
    X = df_ml_ready[colunas_cat + colunas_climaticas]
    y = df_ml_ready['quantidade_produzida']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    c1, c2 = st.columns(2)
    c1.metric("Amostras de Treino (X_train)", f"{X_train.shape[0]:,} linhas")
    c2.metric("Amostras de Teste (X_test)", f"{X_test.shape[0]:,} linhas")

    st.write("Amostra da Matriz de Entrada (Features X_train):")
    st.dataframe(X_train.head(10), use_container_width=True)

    st.write("Amostra do Alvo de Produção (Target y_train):")
    st.dataframe(y_train.head(10), use_container_width=True)

    st.markdown('<h4 style="color: #111827 !important;">Exportação de Dados</h4>', unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            label="Baixar X_train_agricola.csv",
            data=X_train.to_csv(index=False),
            file_name="X_train_agricola.csv",
            mime="text/csv"
        )
    with col_btn2:
        st.download_button(
            label="Baixar y_train_agricola.csv",
            data=y_train.to_csv(index=False),
            file_name="y_train_agricola.csv",
            mime="text/csv"
        )