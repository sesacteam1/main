# 코사인유사도 / # 내용의 흐름 일치 검증

# 개발 환경
# Python 3.11
import logging
from dotenv import load_dotenv
import os
from datetime import datetime
import time


# DB
import pymysql
import faiss
import numpy as np

# ChatBot
import openai
import streamlit as st

import json

# LangChain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


from langchain.text_splitter import RecursiveCharacterTextSplitter
import locale


# 로그 설정
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# MySQL 연결 정보
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

log_print = []

# ✅ 1️⃣ 질문 분류 GPT
def classify_question(question):
    """질문을 분석하여 카테고리를 반환하는 GPT"""
    logging.info(f"🟡 [질문 분류] 입력 질문: {question}")
    log_print.append(f"{timestamp}🟡 [질문 분류] 입력 질문: {question}")

    system_prompt = """
    당신은 사용자의 질문을 적절한 카테고리로 분류하는 AI입니다.
    
    가능한 카테고리:
    1. 채용 정보
    2. 일반 질문
    
    사용자의 질문에 '요청'이 포함되지 않았을시 일반 질문으로 분류하세요.
    질문을 분석하여 가장 적절한 카테고리를 JSON 형식으로 출력하세요.

    **출력 형식 (JSON)**
    {"category": "채용 정보"}
    """

    ####사람들의 로그데이터 분석을 통해서 카테고리를 세분화 하겠다~. 배타적이여야함.

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=50
    )

    result = json.loads(response["choices"][0]["message"]["content"])
    logging.info(f"🟢 [질문 분류 완료] 카테고리: {result['category']}")
    return result  # JSON 변환

# ✅ 2️⃣ 데이터 검색 GPT
def connect_to_db():
    """MySQL 데이터베이스 연결"""
    return pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user=MYSQL_USER,
        passwd=MYSQL_PASSWORD,
        db='sesacteam1',
        charset="utf8mb4"
    )

@st.cache_data
def fetch_from_saramin_refined():
    """MySQL에서 채용 정보를 가져옴"""
    connection = connect_to_db()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM saramin_5_combined_revised_fordb")
    fetch_result = cursor.fetchall()
    connection.close()
    
    logging.info(f"🟢 [DB 데이터 로드 완료] 총 {len(fetch_result)}개 문서 가져옴")
    return fetch_result

@st.cache_data
def convert_to_documents(courses):
    """MySQL 데이터를 LangChain 문서로 변환 (청크 없음)"""
    documents = []
    for course in courses:
        doc_text = "\n".join([f"{key}: {value}" for key, value in course.items()])
        documents.append(Document(page_content=doc_text))  # 청크 없이 전체 문서 저장

    logging.info(f"🟢 [문서 변환 완료] 총 {len(documents)}개 문서 생성됨")
    return documents

def normalize_vector_store(store):
    """FAISS 벡터들을 정규화하여 코사인 유사도 기반 검색이 가능하게 변경"""
    xb = store.index.reconstruct_n(0, store.index.ntotal)  # 모든 벡터 가져오기
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)  # 벡터 정규화
    
    # 정규화된 벡터의 L2 노름 확인
    norm = np.linalg.norm(xb, axis=1)
    if np.allclose(norm, 1.0):
        print("벡터는 모두 정규화되었습니다.")
    else:
        print("벡터가 일부 정규화되지 않았습니다.")

    store.index = faiss.IndexFlatIP(xb.shape[1])  # 내적 기반 인덱스 다시 생성
    store.index.add(xb)  # 정규화된 벡터 추가
    # 새로 생성된 인덱스의 metric_type 출력
    print("새로 생성된 인덱스의 metric_type:", store.index.metric_type)
    return store

@st.cache_resource
def create_vector_store(_documents):
    """벡터 스토어를 불러오거나, 없으면 새로 생성하여 저장"""
    embeddings = OpenAIEmbeddings()
    faiss_path = './db/faiss_norm'  # 벡터 스토어 경로

    # ✅ 벡터 스토어가 존재하면 불러오기
    if os.path.exists(faiss_path):
        logging.info("🟡 [벡터 스토어 로드] 기존 벡터 스토어를 불러옵니다.")
        store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # ❌ 존재하지 않으면 새로 생성 후 저장
        logging.info("🔵 [벡터 스토어 생성] 새로운 벡터 스토어를 만듭니다.")
        store = FAISS.from_documents(documents, embeddings)
        # 🔹 코사인 유사도 적용: 벡터 정규화
        store = normalize_vector_store(store)
        
        
        store.save_local(faiss_path)  # 벡터 스토어 저장
        logging.info("🟢 [벡터 스토어 저장 완료] 새로운 벡터 스토어를 저장했습니다.")

    logging.info(f"🟢 [벡터 스토어 준비 완료] 문서 개수: {len(documents)}")
    
    return store

def create_chatbot(vector_store):
    """채용 정보 기반 챗봇 생성"""
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt_template = PromptTemplate(
        input_variables=["question", "context"],  # 필요한 변수 정의
        template="""
        시스템 프롬프트:
        1. 당신은 잡Job의 채용 전문가입니다.참조 문서를 기반으로 사용자의 요구에 맞는 채용 정보를 제공하세요.
        2. 채용 공고에 대한 정보를 제공할시 다음과 같은 형식으로 답변하세요. '공고제목', '공고링크', '기업명'을 반드시 포함하세요. 
        - 공고 제목: (공고제목)
        - 공고 링크: (공고링크를 그대로 제공하세요)
        - 기업명:(기업명)
        - 태그:(태그)
        - 주요 업무: (주요 업무에 대한 설명)
        - 자격 요건: (필수 자격 요건)
        - 경력: (경력)

        3. 관련 없는 질문에는 자연스럽게 채용 정보 질문을 유도하세요.
        4. 직급과 관련된 답변은 포함하지 마세요.

        질문: {question}
        
        관련 문서:
        {context}
        """
    )

    return ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt_template}  # 이제 PromptTemplate 객체 사용
    )

def get_answer(question, retrieval_chain):
    """채용 정보를 검색하여 응답 생성"""
    result = retrieval_chain.invoke({"question": question, "chat_history": []})

    answer = result.get("answer", "")
    logging.info(f"🟢 [답변 생성 완료] 답변: {answer[:100]}...")  # 너무 길면 자름

    return answer, result.get("source_documents", [])

# ✅ 3️⃣ 질문-답변 검증 GPT
def verify_answer(question, answer):
    """질문과 답변이 논리적으로 맞는지 검증"""
    system_prompt = """
    당신은 AI 검증 시스템입니다. 질문과 답변의 흐름이 논리적으로 연결되는지 판단하세요. 판별을 '일치'로 판별하세요

    - 질문과 답변의 내용 흐름이 연결되면: {"verification": "일치"}
    - 질문과 답변의 내용 흐름이 연결되지 않으면: {"verification": "불일치"}
    """
    ### 불일치 하는 과정을 확인. 불일치 비율 검증. 

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"질문: {question}\n답변: {answer}"}
        ],
        temperature=0.3,
        max_tokens=50
    )

    result = json.loads(response["choices"][0]["message"]["content"])
    logging.info(f"🟢 [답변 검증 완료] 결과: {result['verification']}")
    return result

# ✅ 대화 기록 저장을 위한 리스트
chat_history = []

def get_answer_with_similarity_check(question, vector_store, chat_history):
    """채용 정보를 검색하여 응답을 생성하며, 유사도가 0.2 이상인지 확인"""
    retriever = vector_store.as_retriever()
    print(retriever.vectorstore.index.metric_type)

    
    # 유사도 검사 (최대 3개의 문서 검색)
    docs_with_scores = retriever.vectorstore.similarity_search_with_score(question, k=3)
    
    # 가장 높은 유사도를 가진 문서 찾기
    if not docs_with_scores:
        logging.warning("❌ 관련 문서를 찾지 못했습니다.")
        return "⚠ 관련된 채용 정보를 찾지 못했습니다. 좀 더 구체적으로 질문해 주세요.", []
    
    highest_score = max(score for _, score in docs_with_scores)
    logging.info(f'✅유사도 최고 점수: {highest_score}')

    # ✅ 유사도가 0.3 이상이면 문서를 기반으로 응답 생성
    if highest_score >= 0.3:
        retrieval_chain = create_chatbot(vector_store)
        result = retrieval_chain.invoke({
            "question": question, 
            "chat_history": chat_history  # ✅ 기존 대화 기록 추가
        })
        answer = result.get("answer", "")

        # ✅ 대화 기록 업데이트
        chat_history.append({"question": question, "answer": answer})
        
        return answer, result.get("source_documents", [])

    # ❌ 유사도가 낮을 경우 다시 질문을 유도
    logging.warning("❌ 유사도가 낮아 문서를 참조하지 않습니다.")
    return "⚠ 현재 질문과 관련된 채용 정보를 찾을 수 없습니다. 조금 더 상세히 질문해 주세요!", []


# ✅ 4️⃣ 전체 실행 함수 (대화 기록 활용)
def handle_request(question):
    """질문을 처리하고, 유사도 기준을 적용하여 데이터 검색"""
    
    # 1️⃣ 질문 분류 실행
    category_result = classify_question(question)
    category = category_result["category"]
    
    if category == "일반 질문":
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                "content": """당신은 잡Job의 채용 전문가입니다. 
                            당신은 채용 정보에 관한 질문에만 대답할 수 있습니다. 그 외의 질문에는 한줄의 단답형으로 답하세요.
                            '채용 공고와 관련된 질문이 있으신가요?'를 마지막에 덧붙이세요"""},
                # ✅ 대화 기록 추가
                *[
                    {"role": "user", "content": entry["question"]}
                    for entry in chat_history
                ],
                *[
                    {"role": "assistant", "content": entry["answer"]}
                    for entry in chat_history
                ],
                {"role": "user", "content": question}
            ],
            temperature=0.2,
            max_tokens=300
        )
        
        answer = response["choices"][0]["message"]["content"]
        
        # ✅ 대화 기록 업데이트
        chat_history.append({"question": question, "answer": answer})
        
        return answer

    # 2️⃣ 채용 정보 검색 실행
    courses = fetch_from_saramin_refined()
    documents = convert_to_documents(courses)
    vector_store = create_vector_store(documents)

    # 2️⃣.5️⃣ 유사도 검사 및 응답 생성 (대화 기록 추가)
    answer, source_docs = get_answer_with_similarity_check(question, vector_store, chat_history)

    # 3️⃣ 질문-답변 검증 실행 (유사도 기준 통과한 경우만 검증)
    if source_docs:
        verification_result = verify_answer(question, answer)
        if verification_result["verification"] == "일치":
            return answer  
        else:
        # 검증이 실패했을 경우 사용자에게 다시 질문을 유도하는 메시지 반환
            return "답변이 정확하지 않습니다. 다시 질문해 주세요."
        

    return answer  # 유사도 낮을 경우 질문 유도 메시지 반환

# ✅ 실행 예제
# if __name__ == "__main__":
#     test_question = "식품공학전공 공고 추천."
#     print(handle_request(test_question))



# Streamlit UI
st.set_page_config(page_title="AI Chatbot", layout="wide")

# 사이드바 (로그 저장)
# st.sidebar.title("🔍 검색 로그")

# if "logs" not in st.session_state:
#     st.session_state.logs = []

# ###
# # log_print 리스트가 있다면 세션 상태에 추가
# if "log_print" in locals() and log_print:  
#     st.session_state.logs.extend(log_print)
#     log_print.clear()  # log_print를 초기화하여 중복 저장 방지

# # 사이드바에 로그 출력 (한 줄씩)
# for log in st.session_state.logs:
#     st.sidebar.text(log)
###
###
# for log in st.session_state.logs:
#     st.sidebar.text(log)


###

# 메인 챗 인터페이스
st.title("💬 잡Job 채용 공고 추천 챗봇")
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" in st.session_state:
    # conversation_history가 리스트인지 확인 후 반복
    if isinstance(st.session_state.conversation_history, list):
        for message in st.session_state.conversation_history:
            if isinstance(message, dict) and "assistant" in message:
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    # 사용자의 마지막 메시지에 assistant 응답 추가
                    st.session_state.messages[-1]["content"] += f"\n\n{message['assistant']}"
                else:
                    # 새로운 assistant 메시지 추가
                    st.session_state.messages.append({"role": "assistant", "content": message["assistant"]})



# 이전 대화 불러오기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요...")
if user_input:
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_history.append(f"User: {user_input}")
    with st.chat_message("user"):
        st.markdown(user_input)
    # st.write(st.session_state.messages)
    
    # FAISS 검색 수행
    data = fetch_from_saramin_refined()
    documents = convert_to_documents(data)
    vector_store = create_vector_store(documents)
    chatbot = create_chatbot(vector_store)
    st.session_state["vector_store"] = vector_store
    st.session_state["chatbot"] = chatbot

    # index, ids = build_faiss_index(data)
    # relevant_ids = search_faiss(user_input, index, ids)
    # st.session_state.logs.append(f"🔹 검색된 문서 ID: {relevant_ids}")
    
    # OpenAI 응답 생성
    response_text = handle_request(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.conversation_history.append(f"assistant: {response_text}")
    
    # AI 응답 표시
    with st.chat_message("assistant"):
        st.markdown(response_text)

    #for message in st.session_state.conversation_history:
        # st.write(message)