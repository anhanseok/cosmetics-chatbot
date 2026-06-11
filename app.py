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
LLM_MINI = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # LLM only 비교용

# ==========================================
# 쿼리 확장 딕셔너리
# ==========================================
QUERY_EXPANSION = {
    "틴트": "틴트 립틴트 립스틱 립메이크업 립컬러 립글로스",
    "립": "립스틱 립틴트 틴트 립메이크업 립컬러 립글로스",
    "립메이크업": "립메이크업 립스틱 틴트 립틴트 립컬러 립글로스",
    "쿠션": "쿠션 파운데이션 베이스메이크업 BB크림",
    "파데": "파운데이션 쿠션 베이스메이크업",
    "파운데이션": "파운데이션 쿠션 베이스메이크업",
    "베이스": "베이스메이크업 파운데이션 쿠션 BB크림",
    "베이스메이크업": "베이스메이크업 파운데이션 쿠션 BB크림",
    "선크림": "선크림 선케어 자외선차단 썬크림 썬스크린",
    "썬크림": "썬크림 선크림 선케어 자외선차단",
    "자외선": "자외선차단 선크림 선케어 썬스크린",
    "세럼": "세럼 에센스 앰플 에센스_세럼_앰플",
    "에센스": "에센스 세럼 앰플 에센스_세럼_앰플",
    "앰플": "앰플 세럼 에센스 에센스_세럼_앰플",
    "클렌징폼": "클렌징폼 클렌징 세안 세정",
    "클렌징": "클렌징 클렌징폼 클렌징오일 세안 세정",
    "세안": "세안 클렌징 클렌징폼 클렌징오일",
    "클렌징오일": "클렌징오일 클렌징 오일클렌저 메이크업제거",
    "오일클렌저": "클렌징오일 오일클렌저 클렌징",
    "메이크업제거": "클렌징오일 클렌징 메이크업제거",
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
            params={"query": f"{product_name} 올리브영", "display": 3, "sort": "sim"}
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
            params={"query": product_name, "display": 1, "sort": "sim"}
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
    db = FAISS.load_local(FAISS_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.8},
    )
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4
    hybrid = RunnableParallel(bm25=bm25_retriever, vector=vector_retriever)
    return hybrid | RunnableLambda(rrf_merge)


# ==========================================
# [수정] Top3 추천 + 대화 히스토리 반영
# ==========================================
def generate_answer(question, contexts, history=None):
    context_text = "\n\n".join(contexts)

    # 최근 3턴 히스토리 포맷팅
    history_text = ""
    if history:
        recent = history[-6:]  # user+assistant 쌍 기준 최대 3턴 (6개 메시지)
        turns = []
        for msg in recent:
            role = "사용자" if msg["role"] == "user" else "AI"
            turns.append(f"{role}: {msg['content']}")
        history_text = "\n".join(turns)

    prompt = f"""
너는 올리브영 화장품 리뷰 기반 추천 AI야.
아래 Context를 참고해서 답변해줘.

규칙:
1. Context에 있는 제품 중 질문과 가장 관련 있는 제품 3개를 순위별로 추천해.
2. 각 제품의 추천 근거는 Context의 리뷰 내용을 바탕으로 설명해.
3. 질문의 핵심 키워드(예: 토너, 선크림, 시원한 등)와 관련된 제품만 추천해.
4. Context에 질문과 관련된 제품이 없으면 "관련 제품을 찾지 못했습니다."라고 답해.
5. 사용자가 원하는 제형과 다른 제품은 추천하지 마.
6. 이전 대화가 있으면 맥락을 반영해서 답변해.

[중요] 반드시 아래 형식만 사용해. 다른 형식은 절대 사용하지 마.
"추천 제품 N:" 으로 시작하는 줄과 한칸 띄우고 "추천 이유 N:" 으로 시작하는 줄만 출력해.
앞에 숫자나 bullet을 붙이지 마. 형식 외의 텍스트를 추가하지 마.

출력 형식 (이 형식 그대로):
추천 제품 1: 상품명
추천 이유 1: 리뷰에서 확인된 근거 (1~2문장)

추천 제품 2: 상품명
추천 이유 2: 리뷰에서 확인된 근거 (1~2문장)

추천 제품 3: 상품명
추천 이유 3: 리뷰에서 확인된 근거 (1~2문장)

[이전 대화]
{history_text if history_text else "없음"}

[Context]
{context_text}

[Question]
{question}

[Answer]
"""
    response = LLM.invoke(prompt)
    return response.content


# ==========================================
# [수정] Top3 제품명 파싱
# ==========================================
def extract_product_names(answer):
    """'추천 제품 N: 상품명' 패턴으로 최대 3개 추출"""
    products = []
    for line in answer.split("\n"):
        for i in range(1, 4):
            prefix = f"추천 제품 {i}:"
            if prefix in line:
                name = line.replace(prefix, "").strip()
                if name:
                    products.append(name)
    return products


# ==========================================
# [수정] 후속 질문 감지 → 검색 쿼리 보강
# ==========================================
FOLLOWUP_KEYWORDS = ["2번", "3번", "1번", "그거", "그 제품", "왜 좋아", "왜좋아", "성분", "어때", "차이", "더 알려", "자세히", "비교"]

def build_search_query(question, history):
    """후속 질문이면 히스토리 직전 assistant 답변에서 제품명 추출해 쿼리 보강"""
    is_followup = any(kw in question for kw in FOLLOWUP_KEYWORDS)
    if not is_followup or not history:
        return question

    for msg in reversed(history):
        if msg["role"] == "assistant":
            names = extract_product_names(msg["content"])
            if names:
                # "2번이 왜 좋아?" → 2번 인덱스 제품 우선, 나머지도 포함
                if "2번" in question and len(names) >= 2:
                    primary = names[1]
                elif "3번" in question and len(names) >= 3:
                    primary = names[2]
                elif "1번" in question:
                    primary = names[0]
                else:
                    primary = names[0]
                return f"{primary} {question}"
    return question


# ==========================================
# [수정] Top3 이미지+가격 렌더링
# ==========================================
def render_product_results(answer):
    """Top3 제품 이미지 + 최저가 렌더링"""
    product_names = extract_product_names(answer)
    if not product_names:
        return

    cols = st.columns(len(product_names))
    for idx, (col, product_name) in enumerate(zip(cols, product_names)):
        with col:
            st.markdown(f"**{idx+1}위: {product_name}**")

            img_bytes = get_product_image(product_name)
            if img_bytes:
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, use_container_width=True)
                except:
                    st.caption("이미지 없음")
            else:
                st.caption("이미지 없음")

            shop_info = get_product_shopping_info(product_name)
            if shop_info:
                formatted_price = f"{shop_info['lprice']:,}원"
                st.markdown(f"**최저가:** {formatted_price}")
                st.markdown(f"[🛒 쇼핑 바로가기]({shop_info['link']})")


# ==========================================
# UI 설정
# ==========================================
st.set_page_config(
    page_title="CosMate",
    page_icon="💄",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #fff7fa 0%, #ffffff 100%);
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #2b2b2b;
    margin-bottom: 5px;
}
.sub-title {
    font-size: 17px;
    color: #666;
    margin-bottom: 28px;
}
.product-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #ffe0ea;
    box-shadow: 0 5px 14px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}
.badge {
    display: inline-block;
    background: #fff0f5;
    color: #d6336c;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}
.product-name {
    font-size: 22px;
    font-weight: 800;
    color: #222;
}
.info-text {
    color: #555;
    line-height: 1.7;
}
.stButton > button {
    background: linear-gradient(135deg, #ff7aa2, #ff4f87);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 800;
    height: 48px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff6694, #ff3f7b);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💄 CosMate ⋆ </div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">실제 화장품 리뷰 데이터를 기반으로 개인 맞춤형 제품을 추천하는 AI 뷰티 어시스턴트</div>',
    unsafe_allow_html=True
)

# ==========================================
# 사이드바
# ==========================================
with st.sidebar:
    st.header("피부 정보 입력")

    category = st.radio("카테고리", ["기초제품", "메이크업"], horizontal=True)

    skin_type = ""
    concerns = []
    skin_tone = ""
    detail = ""
    texture = ""

    if category == "기초제품":
        texture = st.selectbox(
            "제품 종류",
            ["상관없음", "선케어", "에센스_세럼_앰플", "클렌징오일", "클렌징폼"]
        )
        skin_type = st.selectbox(
            "피부 타입",
            ["건성", "지성", "복합성", "민감성", "수부지", "잘 모르겠음"]
        )
        concerns = st.multiselect(
            "피부 고민",
            ["여드름", "홍조", "건조함", "피지", "모공", "잡티", "각질", "탄력 저하"]
        )

    else:
        texture = st.selectbox("제품 종류", ["립메이크업", "베이스메이크업"])

        if texture == "립메이크업":
            skin_tone = st.selectbox(
                "피부톤",
                ["쿨톤", "웜톤", "뉴트럴톤", "잘 모르겠음"]
            )
            detail = st.selectbox(
                "세부 정보",
                ["상관없음", "발색", "지속력", "착색", "광택", "촉촉"]
            )

        elif texture == "베이스메이크업":
            skin_tone = st.selectbox(
                "피부톤",
                ["밝은 피부", "중간 피부", "어두운 피부", "잘 모르겠음"]
            )
            detail = st.selectbox(
                "세부 정보",
                ["상관없음", "커버력", "지속력", "매트", "세미매트", "촉촉", "밀착", "모공", "요철"]
            )

    recommend_btn = st.button("✨ 화장품 추천받기", use_container_width=True)


# ==========================================
# 추천 버튼
# ==========================================
st.subheader("✨ 맞춤 화장품 추천 ໒꒱ ‧₊˚")

if not recommend_btn:
    st.info("왼쪽 사이드바에서 피부 정보를 입력한 뒤 추천 버튼을 눌러주세요.")

if recommend_btn:
    if category == "기초제품" and not concerns:
        st.warning("피부 고민을 하나 이상 선택해주세요.")
    else:
        if category == "기초제품":
            concerns_str = ", ".join(concerns)
            query = f"{skin_type} 피부에 {concerns_str} 고민이 있고 {texture} 제품을 원해요. 맞는 제품 추천해줘."
        else:
            query = f"{texture} 제품 중 {skin_tone} 피부톤에 {detail} 특성을 가진 제품 추천해줘."

        with st.spinner("추천 중..."):
            retriever = load_retriever()
            expanded_query = expand_query(query)
            docs = retriever.invoke(expanded_query)
            contexts = [doc.page_content for doc in docs]
            answer = generate_answer(query, contexts)

        with st.container(border=True):
            st.markdown("### 추천 결과")
            st.write(answer)
            st.divider()
            render_product_results(answer)


# ==========================================
# 질문 분류: recommend / consult / vague
# ==========================================
def classify_question(question, history):
    """few-shot LLM으로 recommend / consult / vague 분류"""
    # 후속 질문(2번, 3번 등)은 키워드로 빠르게 처리
    if any(kw in question for kw in FOLLOWUP_KEYWORDS):
        return "recommend"

    examples = """
Q: 촉촉한 선크림 추천해줘 → recommend
Q: 요즘 여드름이 너무 심해져 → consult
Q: 잡티를 해결할 수 있는 기초제품 뭐가 있어 → recommend
Q: 왜 이렇게 피부가 예민해졌지 → consult
Q: 쿨톤인데 립 추천해줘 → recommend
Q: 화장품 추천해줘 → vague
Q: 홍조를 해결할 수 있는 기초제품은 뭐가 있을까 → recommend
Q: 피부가 너무 건조한데 어떡하지 → consult
Q: 지성 피부에 맞는 세럼 있어? → recommend
Q: 트러블이 자꾸 올라와 → consult
Q: 모공에 좋은 클렌징폼 추천 → recommend
Q: 피부가 왜 이렇게 뒤집어졌지 → consult
Q: 안녕 → vague
Q: 그냥 뭐 좋아? → vague
"""

    response = LLM.invoke(f"""아래 예시를 참고해서 질문 유형을 분류해.
{examples}
Q: {question} →

반드시 "recommend", "consult", "vague" 중 하나만 답해. 다른 말은 하지 마.""")

    result = response.content.strip().lower()
    if "recommend" in result:
        return "recommend"
    elif "consult" in result:
        return "consult"
    return "vague"



def generate_consult(question, history):
    """피부 고민 상담 전용 LLM 호출 (RAG 없음)"""
    history_text = ""
    if history:
        recent = history[-6:]
        turns = []
        for msg in recent:
            role = "사용자" if msg["role"] == "user" else "AI"
            turns.append(f"{role}: {msg['content']}")
        history_text = "\n".join(turns)

    response = LLM.invoke(f"""
너는 피부 전문 상담사야.
사용자의 피부 고민을 듣고 공감하면서 조언해줘.

규칙:
1. 제품 추천은 절대 하지 마.
2. 고민의 원인을 1~2문장으로 설명해.
3. 공감하는 말로 시작해.
4. 개선에 도움이 되는 생활습관을 번호 목록으로 5가지 구체적으로 알려줘.

[이전 대화]
{history_text if history_text else "없음"}

[Question]
{question}
""")
    return response.content


# ==========================================
# 순수 LLM 답변 (RAG 없음, 비교용)
# ==========================================
def generate_llm_only(question, q_type, history=None):
    """RAG 없이 gpt-4o-mini 자체 지식으로만 답변 (q_type에 따라 프롬프트 분기)"""
    history_text = ""
    if history:
        recent = history[-6:]
        turns = []
        for msg in recent:
            role = "사용자" if msg["role"] == "user" else "AI"
            turns.append(f"{role}: {msg['content']}")
        history_text = "\n".join(turns)

    if q_type == "recommend":
        prompt = f"""{question}"""

    else:  # consult
        prompt = f"""아래 피부 고민에 1~2문장으로만 짧게 답변해. 제품 추천은 하지 마.

{question}"""

    response = LLM_MINI.invoke(prompt)
    return response.content


# ==========================================
# 챗봇
# ==========================================
st.divider()
st.subheader("💬 피부 상담 챗봇 𖦹°‧")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("화장품이나 피부 고민에 대해 질문해보세요.")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if category == "기초제품":
        concerns_str = ", ".join(concerns) if concerns else "없음"
        full_question = f"[피부타입: {skin_type}, 고민: {concerns_str}, 제품종류: {texture}]\n{question}"
    else:
        full_question = f"[카테고리: {texture}, 피부톤: {skin_tone}, 세부정보: {detail}]\n{question}"

    history = st.session_state.messages[:-1]
    q_type = classify_question(question, history)  # recommend / consult / vague

    with st.spinner("답변 생성 중..."):
        if q_type == "vague":
            rag_answer = (
                "어떤 종류의 화장품을 찾으시나요? 😊\n\n"
                "예) 선크림, 립틴트, 세럼, 클렌징오일 등\n"
                "피부 타입(건성/지성/민감성)이나 피부 고민(건조함/모공/여드름 등)도 알려주시면 더 잘 추천해드릴 수 있어요!"
            )
            llm_answer = None  # vague는 비교 불필요
        elif q_type == "consult":
            rag_answer = generate_consult(question, history)
            llm_answer = generate_llm_only(full_question, q_type="consult", history=history)
        else:  # recommend
            retriever = load_retriever()
            search_query = build_search_query(question, history)
            expanded_question = expand_query(search_query)
            docs = retriever.invoke(expanded_question)
            contexts = [doc.page_content for doc in docs]
            rag_answer = generate_answer(full_question, contexts, history=history)
            llm_answer = generate_llm_only(full_question, q_type="recommend", history=history)

    # 세션에는 RAG 답변 저장
    st.session_state.messages.append({"role": "assistant", "content": rag_answer})

    # ==========================================
    # UI 렌더링: vague면 단일, 나머지는 2컬럼 비교
    # ==========================================
    if llm_answer:
        col_rag, col_llm = st.columns(2)

        with col_rag:
            st.markdown(
                '<div style="background:#3d5a99;color:white;padding:10px 16px;border-radius:8px;font-weight:700;font-size:15px;margin-bottom:10px;">'
                '📚 RAG 답변 | 리뷰 데이터 기반 검색·생성'
                '</div>',
                unsafe_allow_html=True
            )
            st.chat_message("assistant").write(rag_answer)
            if q_type == "recommend":
                render_product_results(rag_answer)

        with col_llm:
            st.markdown(
                '<div style="background:#b5451b;color:white;padding:10px 16px;border-radius:8px;font-weight:700;font-size:15px;margin-bottom:10px;">'
                '🤖 순수 LLM 답변 | 학습 지식만 사용 (문서 검색 없음)'
                '</div>',
                unsafe_allow_html=True
            )
            st.chat_message("assistant").write(llm_answer)
    else:
        st.chat_message("assistant").write(rag_answer)
