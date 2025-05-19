import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# LEITURA E LIMPEZA DOS DADOS
# =========================

# Carrega o dataset principal e remove a coluna de ID, que não é útil para modelagem
df = pd.read_csv('data/cancelamentos.csv')
df = df.drop(columns=['CustomerID'])
df = df.dropna()  # Remove linhas com valores ausentes

# Carrega uma cópia original dos dados para análises posteriores (sem normalização)
df_original = pd.read_csv('data/cancelamentos.csv')
df_original = df_original.drop(columns=['CustomerID'])
df_original = df_original.dropna()

# =========================
# ENCODING DAS VARIÁVEIS CATEGÓRICAS
# =========================

'''
LabelEncoder: converte categorias em números inteiros automaticamente.
Necessário para que modelos de ML possam trabalhar com variáveis categóricas.
'''
label_encoder = LabelEncoder()
for col in ['sexo', 'assinatura', 'duracao_contrato']:
    df[col] = label_encoder.fit_transform(df[col])

# =========================
# NORMALIZAÇÃO DAS VARIÁVEIS NUMÉRICAS
# =========================

scaler = StandardScaler()
colunas_escalonar = [
    'idade', 'tempo_como_cliente', 'frequencia_uso',
    'ligacoes_callcenter', 'dias_atraso', 'total_gasto',
    'meses_ultima_interacao'
]
df[colunas_escalonar] = scaler.fit_transform(df[colunas_escalonar])
df['cancelou'] = df['cancelou'].astype(int)  # Garante que o alvo é inteiro

# =========================
# DIVISÃO EM VARIÁVEIS INDEPENDENTES E DEPENDENTES
# =========================

'''
X: variáveis independentes (features)
Y: variável dependente (alvo)
'''
X = df.drop(columns=['cancelou'])
Y = df['cancelou']

# =========================
# DIVISÃO EM TREINO E TESTE
# =========================

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# =========================
# TREINAMENTO DO MODELO DE REGRESSÃO LOGÍSTICA
# =========================

'''
Treina o modelo de regressão logística para prever o cancelamento.
Avalia o desempenho usando métricas clássicas de classificação.
'''
modelo_logistico = LogisticRegression(random_state=42)
modelo_logistico.fit(X_train, Y_train)
Y_pred_logistico = modelo_logistico.predict(X_test)

print("Relatório de Classificação - Regressão Logística")
print(classification_report(Y_test, Y_pred_logistico))
print("Matriz de Confusão - Regressão Logística")
print(confusion_matrix(Y_test, Y_pred_logistico))
print("Acurácia - Regressão Logística")
print(accuracy_score(Y_test, Y_pred_logistico))

# Mostra a proporção de cancelamentos no dataset
print(df['cancelou'].value_counts(normalize=True))

# =========================
# ANÁLISE EXPLORATÓRIA
# =========================

# Heatmap de correlação entre variáveis
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# Relação entre variáveis e taxa de cancelamento
print(df.groupby('ligacoes_callcenter')['cancelou'].mean())
print(df.groupby('dias_atraso')['cancelou'].mean())
print(df.groupby('idade')['cancelou'].mean())
print(df.groupby('meses_ultima_interacao')['cancelou'].mean())

# =========================
# CLUSTERIZAÇÃO DOS CLIENTES (KMeans)
# =========================

# Agrupa clientes em 8 clusters com base nas features
kmeans = KMeans(n_clusters=8, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
df_original['cluster'] = df['cluster']  # Adiciona cluster ao dataframe original para análise

# Calcula estatísticas médias por cluster (para uso na API)
cluster_stats = df_original.groupby('cluster').mean(numeric_only=True).to_dict()

# Mostra taxa de cancelamento por cluster e médias dos clusters
print(df.groupby('cluster')['cancelou'].mean())
print(df_original.groupby('cluster').mean(numeric_only=True).T)  # Médias reais

# Gráfico: distribuição de cancelamentos por cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='cancelou', data=df)
plt.title('Distribuição de Cancelamentos por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Contagem')
plt.legend(title='Cancelou', loc='upper right')
plt.show()

# =========================
# TREINAMENTO DO MODELO RANDOM FOREST
# =========================

'''
Treina o modelo Random Forest para prever o cancelamento.
Avalia o desempenho usando métricas clássicas de classificação.
'''
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, Y_train)
Y_pred_rf = modelo_rf.predict(X_test)

print("\nRelatório de Classificação - Random Forest")
print(classification_report(Y_test, Y_pred_rf))
print("Matriz de Confusão - Random Forest")
print(confusion_matrix(Y_test, Y_pred_rf))
print("Acurácia - Random Forest")
print(accuracy_score(Y_test, Y_pred_rf))

# =========================
# SALVAMENTO PARA USO NA API
# =========================

joblib.dump(modelo_logistico, 'model/modelo_regressao_logistica.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(kmeans, 'model/kmeans.pkl')
joblib.dump(cluster_stats, 'model/cluster_stats.pkl')
joblib.dump(modelo_rf, 'model/modelo_random_forest.pkl')