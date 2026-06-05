import os
import io
import pickle
import requests as req
import streamlit as st
from PIL import Image

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableParallel, RunnableLambda

# ==========================================
# API 및 LangSmith 환경변수 설정
# ==========================================
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "OliveYoung_Cosmetics_Bot"


FAISS_PATH = "./faiss_db"
DOCS_PATH = "./faiss_db/review_docs.pkl"

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")
LLM = ChatOpenAI(model="gpt-4o", temperature=0)

# ==========================================
# 쿼리 확장 딕셔너리
# ==========================================
QUERY_EXPANSION = {
    "틴트": "틴트 립틴트 립스틱 립메이크업 립컬러",
    "립": "립스틱 립틴트 틴트 립메이크업 립컬러",
    "세럼": "세럼 에센스 앰플 에센스_세럼_앰플",
    "에센스": "에센스 세럼 앰플 에센스_세럼_앰플",
    "앰플": "앰플 세럼 에센스 에센스_세럼_앰플",
    "쿠션": "쿠션 파운데이션 베이스메이크업 BB크림",
    "파데": "파운데이션 쿠션 베이스메이크업 파운데이션",
    "파운데이션": "파운데이션 쿠션 베이스메이크업",
    "베이스": "베이스메이크업 파운데이션 쿠션 BB크림 CC크림",
    "선크림": "선크림 선케어 자외선차단 썬크림 썬스크린",
    "썬크림": "썬크림 선크림 선케어 자외선차단 썬스크린",
    "로션": "로션 에멀전 수분크림",
    "토너": "토너 스킨 화장수",
    "스킨": "스킨 토너 화장수",
    "립메이크업": "립메이크업 립스틱 틴트 립틴트 립컬러 립글로스",
    "베이스메이크업": "베이스메이크업 파운데이션 쿠션 BB크림 CC크림 프라이머",
}

def expand_query(question):
    expanded = question
    for keyword, synonyms in QUERY_EXPANSION.items():
        if keyword in question:
            expanded += f" {synonyms}"
    return expanded


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
        bm25=bm25_retriever,
        vector=vector_retriever
    )

    return hybrid | RunnableLambda(rrf_merge)


def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""
너는 올리브영 화장품 리뷰 기반 추천 AI야.
아래 Context를 참고해서 답변해줘.

규칙:
1. Context에 있는 제품 중 질문과 가장 관련 있는 제품 1개를 추천해.
2. 추천 근거는 Context의 리뷰 내용을 바탕으로 설명해.
3. 답변 첫 문장은 추천 제품명으로 시작해.
4. 질문의 핵심 키워드(예: 토너, 선크림, 시원한 등)와 관련된 제품만 추천해.
5. Context에 질문과 관련된 제품이 없으면 "관련 제품을 찾지 못했습니다."라고 답해.
6. 사용자가 원하는 제형과 다른 제품은 추천하지 마.
7. 답변은 2~3문장으로 작성해.

답변 형식:
추천 제품: 상품명
추천 이유: 리뷰에서 확인된 근거

[Context]
{context_text}

[Question]
{question}

[Answer]
"""
    response = LLM.invoke(prompt)
    return response.content


def extract_product_name(answer):
    for line in answer.split("\n"):
        if "추천 제품:" in line:
            return line.replace("추천 제품:", "").strip()
    return None


# ── UI ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI 화장품 추천 및 상담 챗봇",
    page_icon="💄",
    layout="wide"
)

st.title("💄 AI 화장품 추천 및 상담 챗봇")
st.write("피부 타입과 고민을 입력하면 맞춤형 화장품을 추천하고, 궁금한 점을 상담해줍니다.")

with st.sidebar:
    st.header("피부 정보 입력")
    skin_type = st.selectbox(
        "피부 타입",
        ["건성", "지성", "복합성", "민감성", "수부지", "잘 모르겠음"]
    )
    concerns = st.multiselect(
        "피부 고민",
        ["여드름", "홍조", "건조함", "피지", "모공", "잡티", "각질", "탄력 저하"]
    )
    texture = st.selectbox(
        "선호 제형",
        ["상관없음", "크림", "젤", "로션", "세럼", "립메이크업", "베이스메이크업"]  # 토너/패드 제거, 립/베이스 추가
    )
    recommend_btn = st.button("✨ 화장품 추천받기", use_container_width=True)

# ── 추천 ────────────────────────────────────────────
st.subheader("✨ 맞춤 화장품 추천")
if recommend_btn:
    if not concerns:
        st.warning("피부 고민을 하나 이상 선택해주세요.")
    else:
        concerns_str = ", ".join(concerns)
        query = f"{skin_type} 피부에 {concerns_str} 고민이 있고 {texture} 제형을 원해요. 맞는 제품 추천해줘."

        with st.spinner("추천 중..."):
            retriever = load_retriever()
            expanded_query = expand_query(query)          # 쿼리 확장
            docs = retriever.invoke(expanded_query)       # 확장된 쿼리로 검색
            contexts = [doc.page_content for doc in docs]
            answer = generate_answer(query, contexts)     # GPT엔 원본 쿼리

        with st.container(border=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 추천 결과")
                st.write(answer)
            with col2:
                product_name = extract_product_name(answer)
                if product_name:
                    img_bytes = get_product_image(product_name)
                    if img_bytes:
                        try:
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, width=200)
                        except:
                            pass

                    shop_info = get_product_shopping_info(product_name)
                    if shop_info:
                        formatted_price = f"{shop_info['lprice']:,}원"
                        st.markdown(f"**최저가:** {formatted_price}")
                        st.markdown(f"[🛒 네이버 쇼핑에서 보기]({shop_info['link']})")

# ── 챗봇 ────────────────────────────────────────────
st.divider()
st.subheader("💬 피부 상담 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("화장품이나 피부 고민에 대해 질문해보세요.")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    concerns_str = ", ".join(concerns) if concerns else "없음"
    full_question = f"[피부타입: {skin_type}, 고민: {concerns_str}, 선호제형: {texture}]\n{question}"

    with st.spinner("답변 생성 중..."):
        retriever = load_retriever()
        expanded_question = expand_query(question)              # 순수 질문만 확장
        docs = retriever.invoke(expanded_question)              # 확장된 쿼리로 검색
        contexts = [doc.page_content for doc in docs]
        answer = generate_answer(full_question, contexts)       # GPT엔 피부정보 포함

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    product_name = extract_product_name(answer)
    if product_name:
        img_bytes = get_product_image(product_name)
        if img_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, width=200)
            except:
                pass

        shop_info = get_product_shopping_info(product_name)
        if shop_info:
            formatted_price = f"{shop_info['lprice']:,}원"
            st.markdown(f"**최저가:** {formatted_price} | [🛒 네이버 쇼핑에서 보기]({shop_info['link']})")
