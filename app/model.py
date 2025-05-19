import joblib
import pandas as pd

# Carrega os modelos e objetos necessários para previsão
modelo_logistico = joblib.load('model/modelo_regressao_logistica.pkl')  # Modelo de regressão logística treinado
scaler = joblib.load('model/scaler.pkl')                                # Scaler para normalização dos dados
kmeans = joblib.load('model/kmeans.pkl')                                # Modelo KMeans para clusterização
cluster_stats = joblib.load('model/cluster_stats.pkl')                  # Estatísticas médias dos clusters
random_forest = joblib.load('model/modelo_random_forest.pkl')           # Modelo Random Forest treinado

# Lista das colunas que devem ser normalizadas
colunas_escalonar = [
    'idade', 'tempo_como_cliente', 'frequencia_uso',
    'ligacoes_callcenter', 'dias_atraso', 'total_gasto',
    'meses_ultima_interacao'
]

def preprocessar_dados(dados_dict):
    """
    Recebe um dicionário com os dados do cliente,
    aplica normalização nas colunas numéricas e
    retorna um DataFrame pronto para previsão.
    """
    df = pd.DataFrame([dados_dict])
    df[colunas_escalonar] = scaler.transform(df[colunas_escalonar])
    # Garante que as colunas estejam na mesma ordem do modelo treinado
    df = df[modelo_logistico.feature_names_in_]
    return df

def prever_cancelamento(dados_dict):
    """
    Recebe um dicionário de dados do cliente,
    retorna a previsão (0 = não cancela, 1 = cancela).
    """
    df = preprocessar_dados(dados_dict)
    return int(modelo_logistico.predict(df)[0])

def prever_perfil(dados_dict):
    """
    Recebe um dicionário de dados do cliente,
    retorna o número do cluster (perfil) ao qual pertence.
    """
    df = preprocessar_dados(dados_dict)
    return int(kmeans.predict(df)[0])

def prever_random_forest(dados_dict):
    '''
    Recebe um dicionário de dados do cliente,
    retorna a previsão de cancelamento usando Random Forest.
    '''
    df = preprocessar_dados(dados_dict)
    return int(random_forest.predict(df)[0])

def get_cluster_stats(cluster_num):
    """
    Recebe o número do cluster e retorna as médias das variáveis
    para aquele perfil, como um dicionário.
    """
    return {col: cluster_stats[col][cluster_num] for col in cluster_stats}