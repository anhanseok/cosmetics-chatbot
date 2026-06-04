import os
import io
import pickle
import requests as req
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 환경변수 로드 및 API 키 설정
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

FAISS_PATH = "./faiss_db"
DOCS_PATH = "./faiss_db/review_docs.pkl"

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def get_product_image(product_name):
    client_id = os.environ.get("NAVER_CLIENT_ID")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET")

    try:
        response = req.get(
            "https://openapi.naver.com/v1/search/image",
            headers={
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret,
            },
            params={
                "query": f"{product_name} 올리브영",
                "display": 3,
                "sort": "sim"
            }
        )
        result = response.json()
        if result.get("items"):
            for item in result["items"]:
                img_url = item["link"]
                try:
                    img_response = req.get(img_url, timeout=3)
                    if img_response.status_code == 200:
                        return img_response.content
                except:
                    continue
    except:
        pass
    return None


def get_product_shopping_info(product_name):
    client_id = os.environ.get("NAVER_CLIENT_ID")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET")

    try:
        response = req.get(
            "https://openapi.naver.com/v1/search/shop.json",
            headers={
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret,
            },
            params={
                "query": product_name,
                "display": 1,
                "sort": "sim"
            }
        )
        result = response.json()
        if result.get("items"):
            item = result["items"][0]
            return {
                "lprice": int(item.get("lprice", 0)),
                "link": item.get("link"),
                "title": item.get("title").replace("<b>", "").replace("</b>", "")
            }
    except:
        pass
    return None


def rrf_merge(results, k=60):
    rrf_scores = {}
    rrf_docs = {}
    for retriever_name, doc_list in results.items():
        for rank, doc in enumerate(doc_list):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = 0.0
                rrf_docs[key] = doc
            rrf_scores[key] += 1 / (k + rank + 1)
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [rrf_docs[k] for k in sorted_keys]


@st.cache_resource
def load_retriever():
    db = FAISS.load_local(
        FAISS_PATH,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.8},
    )

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4

    hybrid = RunnableParallel(
        bm25=
