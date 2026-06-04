import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

FAISS_PATH = "./faiss_db"
DOCS_PATH = "./faiss_db/review_docs.pkl"

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)


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
반드시 아래 Context에 있는 내용만 사용해서 답변해.

중요 규칙:
1. Context에 없는 효능, 성분, 피부 타입, 제품 특징은 절대 말하지 마.
2. 질문에서 요구한 키워드와 직접 관련된 리뷰 근거만 사용해.
3. 추천 제품은 1개만 말해.
4. 답변 첫 문장은 질문에 대한 직접적인 추천으로 시작해.
5. 근거는 Context에 있는 표현을 바탕으로 짧게 설명해.
6. 확실한 근거가 없으면 "리뷰만으로는 명확히 추천하기 어렵습니다."라고 답해.
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
        ["상관없음", "크림", "젤", "로션", "세럼", "토너", "패드"]
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
            docs = retriever.invoke(query)
            contexts = [doc.page_content for doc in docs]
            answer = generate_answer(query, contexts)

        with st.container(border=True):
            st.markdown("### 추천 결과")
            st.write(answer)

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
        docs = retriever.invoke(full_question)
        contexts = [doc.page_content for doc in docs]
        answer = generate_answer(full_question, contexts)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
