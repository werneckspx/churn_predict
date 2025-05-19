from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import prever_cancelamento, prever_perfil, get_cluster_stats, prever_random_forest

# Inicializa a aplicação FastAPI
app = FastAPI()

class ClienteInput(BaseModel):
    """
    Modelo de dados de entrada para a API.
    Define os campos esperados para um cliente.
    """
    idade: float
    sexo: str
    tempo_como_cliente: float
    frequencia_uso: float
    ligacoes_callcenter: float
    dias_atraso: float
    assinatura: str
    duracao_contrato: str
    total_gasto: float
    meses_ultima_interacao: float

# Dicionários de mapeamento para converter strings em valores numéricos
MAPA_SEXO = {"female": 0, "male": 1}
MAPA_ASSINATURA = {"basic": 0, "premium": 1, "standard": 2}
MAPA_CONTRATO = {"annual": 0, "monthly": 1, "quarterly": 2}

def converter_entrada(cliente: ClienteInput):
    """
    Converte os campos categóricos do cliente de string para inteiro,
    usando os dicionários de mapeamento.
    Lança uma exceção HTTP 400 se algum valor for inválido.
    """
    dados = cliente.dict()
    try:
        dados['sexo'] = MAPA_SEXO[dados['sexo'].strip().lower()]
        dados['assinatura'] = MAPA_ASSINATURA[dados['assinatura'].strip().lower()]
        dados['duracao_contrato'] = MAPA_CONTRATO[dados['duracao_contrato'].strip().lower()]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Valor inválido para campo categórico: {e}")
    return dados

@app.post("/prever_reg_logistica")
def api_prever_reg_logistica(cliente: ClienteInput):
    """
    Endpoint principal da API.
    Recebe os dados do cliente, faz a conversão dos campos categóricos,
    executa as previsões de cancelamento e perfil, e retorna as médias do cluster.
    Em caso de erro inesperado, retorna HTTP 500.
    """
    try:
        dados = converter_entrada(cliente)
        resultado_cancelamento = prever_cancelamento(dados)
        resultado_perfil = prever_perfil(dados)
        stats = get_cluster_stats(resultado_perfil)
        return {
            "dados_recebidos": cliente.dict(),
            "perfil": resultado_perfil,
            "cancelamento": resultado_cancelamento,
            "media_dados_cluster": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    

@app.post("/prever_random_forest")
def api_prever_random_forest(cliente: ClienteInput):
    """
    Endpoint principal da API.
    Recebe os dados do cliente, faz a conversão dos campos categóricos,
    executa as previsões de cancelamento e perfil, e retorna as médias do cluster.
    Em caso de erro inesperado, retorna HTTP 500.
    """
    try:
        dados = converter_entrada(cliente)
        resultado_cancelamento = prever_random_forest(dados)
        resultado_perfil = prever_perfil(dados)
        stats = get_cluster_stats(resultado_perfil)
        return {
            "dados_recebidos": cliente.dict(),
            "perfil": resultado_perfil,
            "cancelamento": resultado_cancelamento,
            "media_dados_cluster": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")