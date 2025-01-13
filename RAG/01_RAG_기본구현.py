from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex
# SimpleDirectoryReader: 지정된 디렉토리에서 문서를 읽어와 텍스트 데이터를 로드하는 클래스
# GPTVectorStoreIndex: 문서에서 정보를 벡터화하여 저장하고 검색하는 데 사용되는 인덱스 생성 클래스


# 문서 로드
documents = SimpleDirectoryReader(input_files=["./RAG/le_Petit_Prince_본문.pdf"]).load_data()

# 인덱스 생성 (메모리 기반)
index = GPTVectorStoreIndex.from_documents(documents)       # 문서 내용을 벡터화(숫자로 변환)하여 검색에 적합하게 만듦

# 쿼리 엔진 생성
query_engine = index.as_query_engine()

# 질문
res = query_engine.query("어린왕자가 키우는 식물은?")
print(res)


