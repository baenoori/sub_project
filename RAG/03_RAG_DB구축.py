from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
import os


# PDF 파일 로드
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# 텍스트 분할
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# 벡터 스토어 생성  
def create_vectorstore(texts):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embedding=embeddings, persist_directory="./data/chroma_db")
    return vectorstore

# 챗봇 생성
def create_chatbot(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    return qa_chain

# 메인 실행 함수
def main():
    # PDF 경로
    pdf_path = "./RAG/le_Petit_Prince_본문.pdf"
    
    # PDF 로드 및 처리
    documents = load_pdf(pdf_path)
    print("PDF 로드 완료.")
    
    # 텍스트 분할
    texts = split_text(documents)
    print("텍스트 분할 완료.")
    
    # 벡터 스토어 생성
    vectorstore = create_vectorstore(texts)
    print("벡터 스토어 생성 완료.")
    
    # # 챗봇 생성
    chatbot = create_chatbot(vectorstore)
    print("챗봇 생성 완료.")
    
    # # 사용자 입력으로 챗봇과 대화
    # print("챗봇과 대화하세요. 'exit' 입력 시 종료.")
    # chat_history = []
    # while True:
    #     query = input("질문: ")
    #     if query.lower() == "exit":
    #         break
    #     response = chatbot({"question": query, "chat_history": chat_history})
    #     chat_history.append((query, response['answer']))
    #     print("답변:", response['answer'])

if __name__ == "__main__":
    main()



