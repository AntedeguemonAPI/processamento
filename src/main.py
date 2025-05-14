from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from functools import lru_cache
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import ast
import os
import json
from keybert import KeyBERT
from collections import Counter
import requests

# ========== Configurações ==========

UPLOAD_DIR = "./uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_SUMMARIZER = "facebook/bart-large-cnn"
MODEL_EMBEDDINGS = "all-MiniLM-L6-v2"
COLLECTION_NAME = "chamados_semantic_search"

# ========== Inicializações ==========

app = FastAPI()

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

summarizer = pipeline("summarization", model=MODEL_SUMMARIZER)
encoder = SentenceTransformer(MODEL_EMBEDDINGS)
qdrant_client = QdrantClient(":memory:")


kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

# ========== Modelos Pydantic ==========

class TextoEntrada(BaseModel):
    texto: str

# ========== Funções Auxiliares ==========

@lru_cache(maxsize=1000)
def gerar_resumo(texto: str) -> str:
    resultado = summarizer(texto, max_length=150, min_length=40, do_sample=False)
    return resultado[0]['summary_text']

def extrair_tag(texto: str) -> str:
    texto_original_para_log = texto 
    try:
        if isinstance(texto, str) and texto.startswith('[') and texto.endswith(']'):
            lista = ast.literal_eval(texto)
            texto_convertido = ' '.join(map(str, lista))
            print(f"Texto original (string de lista): {texto_original_para_log}")
            print(f"Texto convertido para KeyBERT: '{texto_convertido}'")
            texto = texto_convertido 
        elif isinstance(texto, list): 
            texto_convertido = ' '.join(map(str, texto))
            print(f"Texto original (lista): {texto_original_para_log}")
            print(f"Texto convertido para KeyBERT: '{texto_convertido}'")
            texto = texto_convertido

    except Exception as e:
        print(f"Erro ao interpretar texto '{texto_original_para_log}' como lista ou ao processá-lo: {e}")
        return "" 

    if not isinstance(texto, str) or not texto.strip():
        print(f"Texto para KeyBERT está vazio, não é string ou só contém espaços após o pré-processamento. Texto original: {texto_original_para_log}")
        return ""

    keywords = kw_model.extract_keywords(
        texto,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,  # Desabilita a remoção de stop words
        use_mmr=False,  
        top_n=3
    )
    # ---------------------------

    # Adicionar um log para ver o que KeyBERT retorna
    print(f"KeyBERT retornou para '{texto}': {keywords}")

    if keywords:
        return keywords[0][0]
    else:

        print(f"Nenhuma keyword encontrada por KeyBERT para o texto: '{texto}'")
        return ""

def calcular_tempo_resposta(df_linha) -> str:
    try:
        data_inicio = pd.to_datetime(df_linha.get("Data de abertura"))
        data_resposta = pd.to_datetime(df_linha.get("Data de fechamento"))

        if pd.notna(data_inicio) and pd.notna(data_resposta):
            tempo_resposta = (data_resposta - data_inicio).total_seconds() / 3600
            if tempo_resposta < 0:
                return "Chamado ainda não finalizado"
            return tempo_resposta
        else:
            return "Chamado ainda não finalizado"
    except Exception as e:
        print(f"Erro ao calcular tempo de resposta: {e}")
        return None

def indexar_textos(df: pd.DataFrame):
    textos = df["Descrição_tokens_filtered"].astype(str).tolist()
    embeddings = encoder.encode(textos, show_progress_bar=True).astype("float32")

    vector_size = embeddings.shape[1]

    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    pontos = []
    for i in range(len(textos)):
        descricao = str(df.iloc[i]["Descrição"])
        resposta = str(df.iloc[i]["Solução - Solução"]) if "Solução - Solução" in df.columns else None
        categoria = str(df.iloc[i].get("categoria", ""))

        data_abertura = str(df.iloc[i].get("Data de abertura", ""))
        data_fechamento = str(df.iloc[i].get("Data de fechamento", ""))
        id_chamado = str(df.iloc[i].get("ID_chamado", ""))
        data = str(df.iloc[i].get("data", ""))

        print("Texto: ", descricao)
        assunto = extrair_tag(descricao)
        print("Assunto extraído:", assunto)
        tempo_resposta = calcular_tempo_resposta(df.iloc[i])

        ponto = PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "descricao": descricao,
                "resposta_sugerida": resposta,
                "id_chamado": id_chamado,
                "categoria": categoria,
                "data": data,
                "data_abertura": data_abertura,
                "data_fechamento": data_fechamento,
                "tag_assunto": assunto,
                "tempo_resposta_horas": tempo_resposta
            }
        )
        pontos.append(ponto)

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=pontos)

def buscar_similares(consulta: str, filtros=None, top_k=5):
    emb = encoder.encode([consulta]).astype("float32")[0]

    search_filter = None
    if filtros:
        search_filter = {"must": []}
        for campo, valor in filtros.items():
            search_filter["must"].append({
                "key": campo,
                "match": {"value": valor}
            })

    resultados = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=emb,
        limit=top_k,
        query_filter=search_filter
    )

    return [
        {
            "descricao": " ".join(ponto.payload.get("descricao", "").replace("[", "").replace("]", "").replace("'", "").split()).strip(),
            "resposta_sugerida": ponto.payload.get("resposta_sugerida"),
            "tempo_resposta_horas": ponto.payload.get("tempo_resposta_horas"),
            "id_chamado": ponto.payload.get("id_chamado"),
            "categoria": ponto.payload.get("categoria"),
            "data_abertura": ponto.payload.get("data_abertura"),
            "tag_assunto": ponto.payload.get("tag_assunto"),
            "data_fechamento": ponto.payload.get("data_fechamento"),
            "score": ponto.score
        }
        for ponto in resultados
    ]

# ========== Rotas ==========

@app.post("/sumarizar")
async def sumarizar_texto(entrada: TextoEntrada):
    texto = entrada.texto.strip()

    if len(texto.split()) < 10:
        raise HTTPException(status_code=400, detail="Texto muito curto para sumarizar.")
    if len(texto.split()) > 1000:
        raise HTTPException(status_code=400, detail="Texto muito longo para sumarizar (máximo 1000 palavras).")

    resumo = gerar_resumo(texto)
    return {"resumo": resumo}

@app.post("/indexar/{id}")
async def indexar_arquivo(id: str):
    try:
        url = f"http://localhost:5003/texto_limpo/id_geral/{id}"
        response = requests.get(url, timeout=100)

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Erro ao buscar o arquivo JSON de {url}.")

        data = response.json()

        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, (list, tuple)):
                    raise HTTPException(status_code=400, detail=f"Valor de '{key}' não é iterável.")

            data = [dict(zip(data, t)) for t in zip(*data.values())]

        df = pd.DataFrame(data)

        if "Descrição_tokens_filtered" not in df.columns:
            raise HTTPException(status_code=400, detail="Coluna 'Descrição_tokens_filtered' não encontrada no arquivo.")

        indexar_textos(df)

        return {"message": f"{len(df)} interações indexadas com sucesso!"}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erro ao fazer a requisição HTTP: {str(e)}")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar o JSON ou criar o DataFrame: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado: {str(e)}")

@app.get("/busca-semantica")
async def busca_semantica(
    query: str = Query(...),
    palavras_chave: str = Query(None),
    categoria: str = Query(None),
    data: str = Query(None),
    top_k: int = Query(5)
):
    filtros = {}
    if palavras_chave:
        filtros["descricao"] = palavras_chave
    if categoria:
        filtros["categoria"] = categoria
    if data:
        filtros["data"] = data

    resultados = buscar_similares(query, filtros if filtros else None, top_k=top_k)
    return {"resultados": resultados}

@app.get("/top-topics")
async def get_top_topics(
    top_n: int = Query(10, ge=1, description="Número de tópicos mais frequentes a retornar (ex: 10, 25, 50, 100)")
):
    """
    Retorna os tópicos (tags de assunto) mais frequentes na coleção.
    """
    all_tags = []
    offset = None  # Para o scroll da Qdrant

    try:
        while True:

            points, next_offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=250,  
                offset=offset,
                with_payload=["tag_assunto"],
                with_vectors=False
            )

            for point in points:
                tag = point.payload.get("tag_assunto")
                if tag and isinstance(tag, str) and tag.strip():
                    all_tags.append(tag)

            if not next_offset: 
                break
            offset = next_offset

        if not all_tags:
            return {"message": "Nenhum tópico encontrado ou a coleção está vazia.", "top_topics": []}

        tag_counts = Counter(all_tags)
        most_common_tags = tag_counts.most_common(top_n)

        formatted_top_topics = [
            {"topic": tag, "count": count} for tag, count in most_common_tags
        ]

        return {
            "total_documents_with_tags": len(all_tags),
            "total_unique_topics": len(tag_counts),
            "requested_top_n": top_n,
            "top_topics": formatted_top_topics
        }

    except Exception as e:

        print(f"Erro ao buscar tópicos do Qdrant: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar os tópicos: {str(e)}")
