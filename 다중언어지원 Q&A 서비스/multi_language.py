from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
import os
import streamlit as st
from langdetect import detect
import openai
st.set_page_config(page_title="다중 언어 지원 Q&A 서비스", page_icon='💬')

st.title('💬 다중 언어 지원 Q&A 서비스')

# PDF 파일 로드
def load_pdf(uploaded_file):
    # Streamlit 파일 객체를 임시 파일로 저장
    temp_file_path = "./temp_uploaded_file.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # PyMuPDFLoader로 로드
    loader = PyMuPDFLoader(temp_file_path)
    documents = loader.load()
    return documents

# 텍스트 분할
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# 기존 VectorDB 로드 및 새로운 데이터 병합
def update_vectorstore(new_texts):
    persist_directory = "./다중언어지원_QNA_서비스/chroma_db"
    embeddings = OpenAIEmbeddings()

    # 기존 데이터 로드
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("기존 VectorDB 로드 완료.")
    else:
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
        print("새로운 VectorDB 생성.")

    # 새로운 데이터 추가
    vectorstore.add_documents(new_texts)
    print("새로운 데이터 추가 완료.")

    # VectorDB 저장
    vectorstore.persist()
    print("VectorDB 저장 완료.")
    return vectorstore

# 챗봇 생성
def create_chatbot(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    return qa_chain

def translate_some_to_en(text, input_language):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Translate the following text to english:\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        translation = response['choices'][0]['message']['content'].strip()
        return translation
    except Exception as e:
        st.error(f"번역 중 오류가 발생했습니다: {e}")
        return None

def translate_en_to_some(text, input_language):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Translate the following text to {input_language}:\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        translation = response['choices'][0]['message']['content'].strip()
        return translation
    except Exception as e:
        st.error(f"번역 중 오류가 발생했습니다: {e}")
        return None

# 메인 실행 함수
def main():
    language_map = {
        "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
        "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
        "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
        "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
        "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
        "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese",
        "kn": "Kannada", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
        "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali",
        "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish",
        "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sk": "Slovak",
        "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sv": "Swedish",
        "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
        "tl": "Tagalog", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
        "vi": "Vietnamese", "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)"
    }
    
    # PDF 업로드
    pdf_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
    if pdf_file:
        # PDF 로드 및 처리
        documents = load_pdf(pdf_file)
        texts = split_text(documents)
        vectorstore = update_vectorstore(texts)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory="./다중언어지원_QNA_서비스/chroma_db", embedding_function=embeddings)

    # 챗봇 생성
    chatbot = create_chatbot(vectorstore)
    chat_history = []  # 대화 기록 저장

    # 사용자 입력
    user_input = st.text_input("사용자 질문:")
    
    if st.button("질문하기") and user_input:
        # 언어 감지 및 언어 코드 확인
        language_code = detect(user_input)
        input_language_name = language_map.get(language_code, "알 수 없는 언어")
        
        # 입력 텍스트 영어로 번역
        translated_query = translate_some_to_en(user_input, input_language_name)
        response = chatbot({"question": translated_query, "chat_history": chat_history})
        
        # 영어 답변을 원래 언어로 번역
        translated_response = translate_en_to_some(response['answer'], input_language_name).replace("<|eot_id|>", "")
        
        # 대화 기록 업데이트
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": translated_response})
        
        # 출력
        st.write("💬 질문:", user_input)
        st.write("🤖 답변:", translated_response)

if __name__ == "__main__":
    main()