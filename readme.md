# Churn Predict API

[![Build Status](https://img.shields.io/github/actions/workflow/status/SEU_USUARIO/SEU_REPOSITORIO/ci.yml?branch=main)](https://github.com/SEU_USUARIO/SEU_REPOSITORIO/actions)
[![Coverage](https://img.shields.io/codecov/c/github/SEU_USUARIO/SEU_REPOSITORIO)](https://codecov.io/gh/SEU_USUARIO/SEU_REPOSITORIO)
[![License](https://img.shields.io/github/license/SEU_USUARIO/SEU_REPOSITORIO)](LICENSE)
<!-- [![PyPI version](https://img.shields.io/pypi/v/seu-pacote.svg)](https://pypi.org/project/seu-pacote/) -->

---

## Sumário

- [Visão Geral](#visão-geral)
- [Links Rápidos](#links-rápidos)
- [Como rodar](#como-rodar)
- [Como obter os dados](#como-obter-os-dados)
- [Exemplo de uso](#exemplo-de-uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como funciona o pipeline de dados e análise](#como-funciona-o-pipeline-de-dados-e-análise)
- [Observações](#observações)
- [Principais observações analíticas](#principais-observações-analíticas)
- [Changelog](#changelog)
- [Licença](#licença)
- [Contribuição](#contribuição)

---

## Visão Geral

Este projeto é uma API para previsão de cancelamento de clientes (churn) e análise de perfil, utilizando modelos de Machine Learning treinados em Python.

---

## Links Rápidos

- [Documentação interativa (Swagger)](http://localhost:8000/docs)
- [Dataset no Kaggle](https://www.kaggle.com/datasets/adrianosantosdev/dados-de-cancelamento-de-contrato-do-cliente)
- [Como contribuir](#contribuição)

---

## Screenshot

![Swagger UI Screenshot](docs/image.png)
<!-- Adicione um screenshot do Swagger ou um gif curto mostrando a API em ação -->

---

## Como rodar

1. **Clone o repositório e instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Treine os modelos (opcional, se já existirem os arquivos `.pkl` em `/model`):**
    ```bash
    python app/train.py
    ```

3. **Inicie a API:**
    ```bash
    uvicorn app.main:app --reload
    ```

4. **Acesse a documentação interativa:**
    [http://localhost:8000/docs](http://localhost:8000/docs)

## Exemplo de uso

Faça uma requisição POST para `/prever` com o seguinte JSON:

```json
{
  "idade": 30,
  "sexo": "Female",
  "tempo_como_cliente": 39,
  "frequencia_uso": 14,
  "ligacoes_callcenter": 5,
  "dias_atraso": 18,
  "assinatura": "Standard",
  "duracao_contrato": "Annual",
  "total_gasto": 932,
  "meses_ultima_interacao": 17
}
```

## Estrutura do Projeto

```
churn-predict/
│
├── app/
│   ├── main.py
│   ├── model.py
│   └── train.py
├── model/
│   ├── modelo_regressao_logistica.pkl
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   └── cluster_stats.pkl
├── data/
│   └── cancelamentos.csv
├── requirements.txt
└── README.md
```
## Como obter os dados

O arquivo de dados `cancelamentos.csv` **não está neste repositório** devido ao seu tamanho.  
Para rodar o projeto, siga os passos abaixo para baixar o dataset original do Kaggle:

1. Acesse o link do dataset:  
   [https://www.kaggle.com/datasets/adrianosantosdev/dados-de-cancelamento-de-contrato-do-cliente](https://www.kaggle.com/datasets/adrianosantosdev/dados-de-cancelamento-de-contrato-do-cliente)
2. Baixe o arquivo `cancelamentos.csv` e coloque-o na pasta `data/` do projeto.


## Observações

- Os modelos são treinados a partir do arquivo `data/cancelamentos.csv`.
- Os arquivos `.pkl` são salvos na pasta `model/`.
- Para produção, configure variáveis de ambiente e segurança conforme necessário.

---

## Como funciona o pipeline de dados e análise

### 1. Leitura e preparação dos dados
- O arquivo `cancelamentos.csv` é carregado e a coluna `CustomerID` é removida, pois não é útil para previsão.
- Linhas com valores ausentes são descartadas para garantir a qualidade dos dados.

### 2. Transformação de variáveis categóricas
- As colunas `sexo`, `assinatura` e `duracao_contrato` são convertidas de texto para números usando o `LabelEncoder`.  
  Isso é necessário porque modelos de machine learning trabalham apenas com números.

### 3. Normalização dos dados
- As colunas numéricas (como idade, tempo como cliente, total gasto, etc.) são normalizadas com o `StandardScaler`.  
  Isso faz com que todas fiquem na mesma escala, evitando que variáveis com valores maiores dominem o modelo.

### 4. Divisão em treino e teste
- Os dados são divididos em duas partes:  
  - **Treino:** para o modelo aprender.
  - **Teste:** para avaliar se o modelo aprendeu de verdade ou só decorou os dados.

### 5. Treinamento dos modelos
- Um modelo de **Regressão Logística** é treinado para prever se o cliente vai cancelar ou não.
- Um modelo de **Random Forest** também é treinado para a mesma tarefa, permitindo comparar um modelo linear (Regressão Logística) com um modelo mais robusto e não-linear (Random Forest).
- O desempenho de ambos os modelos é avaliado usando métricas como acurácia, precisão, recall, f1-score e matriz de confusão.

### 6. Clusterização (KMeans)
- O algoritmo KMeans agrupa os clientes em 8 perfis (clusters) com características semelhantes.
- Para cada cluster, são calculadas as médias das variáveis, ajudando a entender o perfil típico de cada grupo.
- Um gráfico mostra a distribuição de cancelamentos em cada cluster.

---

## Observações

- O **Random Forest** apresentou acurácia significativamente maior (por exemplo, 0.99) em relação à Regressão Logística (0.85) nos testes.
- Isso pode indicar que o Random Forest está capturando padrões mais complexos nos dados, mas também pode ser um sinal de overfitting, especialmente se o conjunto de teste não for totalmente independente ou se houver vazamento de dados.
- É importante analisar não só a acurácia, mas também a matriz de confusão, precisão, recall e f1-score para garantir que o modelo está generalizando bem e não apenas "decorando" os dados.

---

## Outras observações importantes

- **Pipeline seguro:** O pré-processamento (normalização, encoding) é feito corretamente apenas nos dados de treino antes de ser aplicado nos dados de teste, evitando vazamento de dados.
- **Reprodutibilidade:** O uso de `random_state` garante que os resultados sejam reproduzíveis.
- **Salvamento dos modelos:** Tanto o modelo de regressão logística quanto o Random Forest, além do scaler e do KMeans, são salvos em arquivos `.pkl` para uso posterior na API.
- **API flexível:** A API permite escolher qual modelo usar para previsão, facilitando experimentação e comparação de resultados.

---

### 7. Análise exploratória dos dados
- São gerados gráficos e tabelas para entender melhor os dados, como:
  - Proporção de cancelamentos.
  - Correlação entre variáveis.
  - Relação entre variáveis e a taxa de cancelamento.

---

## Principais observações analíticas

- **Distribuição de Cancelamentos:**  
  Aproximadamente 57% dos clientes cancelaram, indicando leve desbalanceamento.
- **Correlação:**  
  O heatmap de correlação e agrupamentos por variáveis mostram que frequência de uso, dias de atraso e ligações ao call center têm impacto relevante no churn.
- **Clusters:**  
  Os 8 clusters identificados pelo KMeans apresentam perfis distintos, com alguns grupos tendo taxas de cancelamento próximas de 100% e outros bem menores, permitindo insights para retenção.

---